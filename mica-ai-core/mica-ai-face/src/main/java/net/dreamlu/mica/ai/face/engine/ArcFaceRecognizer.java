/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.engine;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.config.FaceConfig;
import net.dreamlu.mica.ai.face.config.FaceEmbedding;
import net.dreamlu.mica.ai.face.preprocess.ImageUtils;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.nio.FloatBuffer;

/**
 * ArcFace 人脸识别引擎（纯 ONNX Runtime）。
 *
 * <p>对应 buffalo_l 模型集中的 {@code w600k_r50.onnx}，输入 112x112 BGR 归一化图像。
 * 输出 512 维 embedding，本引擎在外部再做 L2 归一化。
 *
 * @since 1.0.0
 */
@Slf4j
public class ArcFaceRecognizer implements Closeable, AutoCloseable {

	/** ArcFace 标准 5 关键点参考位置（112x112 坐标系下） */
	private static final float[][] ARC_FACE_REFERENCE = {
		{38.2946f, 51.6963f},   // 左眼
		{73.5318f, 51.5014f},   // 右眼
		{56.0252f, 71.7366f},   // 鼻尖
		{41.5493f, 92.3655f},   // 左嘴角
		{70.7299f, 92.2041f},   // 右嘴角
	};

	private final OrtSession session;
	private final FaceConfig config;

	public ArcFaceRecognizer(FaceConfig config) {
		this.config = config;
		try {
			OrtEnvironment env = OrtEnvironment.getEnvironment();
			OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
			opts.setIntraOpNumThreads(config.getIntraOpNumThreads());
			opts.setInterOpNumThreads(config.getInterOpNumThreads());
			this.session = env.createSession(config.getRecModelPath().toString(), opts);
			log.info("[mica-ai-face] ArcFace loaded: {}", config.getRecModelPath());
		} catch (OrtException e) {
			throw new MicaAiException("Failed to load ArcFace model: " + config.getRecModelPath(), e);
		}
	}

	/**
	 * 给定原图 + 检测到的人脸框（含 5 关键点），裁剪并对齐到 112x112，再做识别。
	 */
	public FaceEmbedding recognize(BufferedImage image, net.dreamlu.mica.ai.face.config.FaceBox box) {
		BufferedImage aligned = align(image, box);
		float[] emb = infer(aligned);
		return new FaceEmbedding(emb);
	}

	private float[] infer(BufferedImage img) {
		try {
			BufferedImage bgr = new BufferedImage(config.getRecInputSize(), config.getRecInputSize(),
				BufferedImage.TYPE_3BYTE_BGR);
			bgr.getGraphics().drawImage(img, 0, 0, null);
			((Graphics2D) bgr.getGraphics()).dispose();

			float[] data = ImageUtils.toNchwFloat(bgr);
			OnnxTensor tensor = OnnxTensor.createTensor(
				OrtEnvironment.getEnvironment(),
				FloatBuffer.wrap(data),
				new long[]{1, 3, config.getRecInputSize(), config.getRecInputSize()}
			);
			try (OrtSession.Result result = session.run(java.util.Collections.singletonMap("input.1", tensor))) {
				OnnxTensor out = (OnnxTensor) result.get(0);
				float[][] vec = (float[][]) out.getValue();
				return l2Normalize(vec[0]);
			} finally {
				tensor.close();
			}
		} catch (OrtException e) {
			throw new MicaAiException("ArcFace inference failed", e);
		}
	}

	/**
	 * ArcFace 标准 5 关键点仿射对齐。
	 */
	private BufferedImage align(BufferedImage src, net.dreamlu.mica.ai.face.config.FaceBox box) {
		float[] lmk = box.getLandmarks();
		if (lmk == null || lmk.length != 10) {
			// 没有关键点时退化为方形裁剪
			return cropCenter(src, box);
		}
		// 求解仿射矩阵 src[3 点] -> dst[3 点]
		// 这里用 5 关键点解仿射略复杂，简化为最小二乘近似的单应估计：用 3 个点
		float[] srcPts = new float[6];
		float[] dstPts = new float[6];
		// 取 左眼(0,1) 右眼(2,3) 鼻尖(4,5) 三个点
		System.arraycopy(lmk, 0, srcPts, 0, 6);
		dstPts[0] = ARC_FACE_REFERENCE[0][0];
		dstPts[1] = ARC_FACE_REFERENCE[0][1];
		dstPts[2] = ARC_FACE_REFERENCE[1][0];
		dstPts[3] = ARC_FACE_REFERENCE[1][1];
		dstPts[4] = ARC_FACE_REFERENCE[2][0];
		dstPts[5] = ARC_FACE_REFERENCE[2][1];
		double[] matrix = solveAffine(srcPts, dstPts);

		// 应用仿射变换
		java.awt.geom.AffineTransform at = new java.awt.geom.AffineTransform(matrix);
		int size = config.getRecInputSize();
		BufferedImage aligned = new BufferedImage(size, size, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = aligned.createGraphics();
		try {
			g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
			g.setColor(new java.awt.Color(0, 0, 0));
			g.fillRect(0, 0, size, size);
			g.setTransform(at);
			g.drawImage(src, 0, 0, null);
		} finally {
			g.dispose();
		}
		return aligned;
	}

	private BufferedImage cropCenter(BufferedImage src, net.dreamlu.mica.ai.face.config.FaceBox box) {
		int x = Math.max(0, (int) box.getX1());
		int y = Math.max(0, (int) box.getY1());
		int w = Math.min(src.getWidth() - x, (int) box.width());
		int h = Math.min(src.getHeight() - y, (int) box.height());
		if (w <= 0 || h <= 0) {
			throw new MicaAiException("Invalid face box: " + box);
		}
		BufferedImage crop = src.getSubimage(x, y, w, h);
		BufferedImage resized = new BufferedImage(config.getRecInputSize(), config.getRecInputSize(),
			BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = resized.createGraphics();
		try {
			g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			g.drawImage(crop, 0, 0, resized.getWidth(), resized.getHeight(), null);
		} finally {
			g.dispose();
		}
		return resized;
	}

	/**
	 * 最小二乘解仿射变换 [a, b, c; d, e, f] 满足 dst = M * src。
	 * src / dst 都是 6 长度数组：[x0, y0, x1, y1, x2, y2]。
	 */
	private double[] solveAffine(float[] src, float[] dst) {
		double sx0 = src[0], sy0 = src[1];
		double sx1 = src[2], sy1 = src[3];
		double sx2 = src[4], sy2 = src[5];
		double dx0 = dst[0], dy0 = dst[1];
		double dx1 = dst[2], dy1 = dst[3];
		double dx2 = dst[4], dy2 = dst[5];

		double det = sx0 * (sy1 - sy2) - sy0 * (sx1 - sx2) + (sx1 * sy2 - sx2 * sy1);
		if (Math.abs(det) < 1e-9) {
			// 退化时返回单位变换
			return new double[]{1, 0, 0, 0, 1, 0};
		}
		double a = (dx0 * (sy1 - sy2) - dy0 * (sx1 - sx2) + (dx1 * sy2 - dx2 * sy1)) / det;
		double b = -(sy0 * (dx1 - dx2) - sy1 * (dx0 - dx2) + (sx0 * dx2 - sx1 * dx1)) / det;
		double c = (sx0 * (sy1 * dx2 - sy2 * dx1) - sy0 * (sx1 * dx2 - sx2 * dx1) + dx0 * (sx1 * sy2 - sx2 * sy1)) / det;
		double d = (dy0 * (sx1 - sx2) - dy1 * (sx0 - sx2) + (sy0 * sx2 - sy1 * sx0)) / det;
		double e = -(sy0 * (dx1 - dx2) - sy1 * (dx0 - dx2) + (sy0 * dx2 - sy1 * dx0)) / det;
		double f = (sx0 * (sy1 * dx2 - sy2 * dx1) - sx0 * (sy1 * dy2 - sy2 * dy1) + (sx0 * sy1 - sx1 * sy0) * dx0) / det;
		return new double[]{a, b, c, d, e, f};
	}

	/**
	 * L2 归一化向量。
	 */
	public static float[] l2Normalize(float[] v) {
		double sum = 0;
		for (float f : v) {
			sum += (double) f * f;
		}
		float norm = (float) Math.sqrt(sum);
		if (norm == 0) {
			return v;
		}
		float[] out = new float[v.length];
		for (int i = 0; i < v.length; i++) {
			out[i] = v[i] / norm;
		}
		return out;
	}

	/**
	 * 计算两个归一化向量的余弦相似度 = 点积。
	 */
	public static float cosineSimilarity(float[] a, float[] b) {
		if (a.length != b.length) {
			throw new MicaAiException("Embedding length mismatch: " + a.length + " vs " + b.length);
		}
		float sum = 0;
		for (int i = 0; i < a.length; i++) {
			sum += a[i] * b[i];
		}
		return sum;
	}

	@Override
	public void close() {
		try {
			session.close();
		} catch (OrtException e) {
			log.warn("[mica-ai-face] failed to close ArcFace session", e);
		}
	}
}
