/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.utils;

import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.config.FaceBox;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

/**
 * 图像工具：读取 / 缩放 / Letterbox / BGR 转换 / 5 关键点仿射对齐。
 *
 * <p>零外部依赖，只用 JDK 自带的 {@link BufferedImage}，避免引入 OpenCV。
 *
 * @since 1.0.0
 */
@UtilityClass
public class ImageUtils {

	/**
	 * 从路径读取图像，自动识别 JPEG/PNG/GIF/BMP。
	 */
	public static BufferedImage read(Path path) throws IOException {
		try (InputStream in = Files.newInputStream(path)) {
			return ImageIO.read(in);
		}
	}

	/**
	 * Letterbox 缩放：保持长宽比，短的边补灰边，结果边长 = {@code size}。
	 *
	 * <p>返回 newImage、缩放比例 scale、原图相对 newImage 的左上角偏移 (dx, dy)。
	 */
	public static LetterboxResult letterbox(BufferedImage src, int size) {
		int sw = src.getWidth();
		int sh = src.getHeight();
		float scale = Math.min((float) size / sw, (float) size / sh);
		int nw = Math.round(sw * scale);
		int nh = Math.round(sh * scale);
		int dx = (size - nw) / 2;
		int dy = (size - nh) / 2;

		BufferedImage dst = new BufferedImage(size, size, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = dst.createGraphics();
		try {
			g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
			g.setColor(new java.awt.Color(114, 114, 114));  // 灰色 padding
			g.fillRect(0, 0, size, size);
			g.drawImage(src, dx, dy, nw, nh, null);
		} finally {
			g.dispose();
		}
		return new LetterboxResult(dst, scale, dx, dy, nw, nh);
	}

	/**
	 * 把 {@link BufferedImage} 转成 ONNX 输入需要的 float32 CHW 张量，RGB 顺序。
	 *
	 * <p>默认会做 (x - 127.5) / 128 归一化（适用于 SFace / ArcFace）。
	 * 如果模型不需要归一化（YuNet），请使用 {@link #toNchwFloat(BufferedImage, boolean)} 重载并传 {@code false}。
	 *
	 * <p>输入图像假定为 {@code TYPE_3BYTE_BGR}，按 BGR 字节序读出，再交换为 RGB。
	 */
	public static float[] toNchwFloat(BufferedImage img) {
		return toNchwFloat(img, true);
	}

	/**
	 * 把 {@link BufferedImage} 转成 float32 CHW 张量（RGB 顺序），可选是否归一化。
	 *
	 * @param img       BGR 格式 BufferedImage
	 * @param normalize true: 做 (x - 127.5) / 128 归一化（SFace / ArcFace）；
	 *                  false: 保留原始 0~255 像素（YuNet）
	 */
	public static float[] toNchwFloat(BufferedImage img, boolean normalize) {
		int w = img.getWidth();
		int h = img.getHeight();
		byte[] raw = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
		float[] out = new float[1 * 3 * h * w];
		int planeSize = h * w;
		for (int y = 0; y < h; y++) {
			int rowStart = y * w * 3;
			for (int x = 0; x < w; x++) {
				int px = rowStart + x * 3;
				float r = raw[px + 2] & 0xFF;
				float g = raw[px + 1] & 0xFF;
				float b = raw[px] & 0xFF;
				if (normalize) {
					r = (r - 127.5f) / 128.0f;
					g = (g - 127.5f) / 128.0f;
					b = (b - 127.5f) / 128.0f;
				}
				out[x + y * w] = r;                       // R plane
				out[planeSize + x + y * w] = g;           // G plane
				out[2 * planeSize + x + y * w] = b;       // B plane
			}
		}
		return out;
	}

	/**
	 * 把检测到的框从 letterbox 坐标系还原到原图坐标系。
	 */
	public static List<FaceBox> mapBoxesBackToOriginal(List<FaceBox> boxes, LetterboxResult lb) {
		List<FaceBox> out = new ArrayList<>(boxes.size());
		for (FaceBox b : boxes) {
			float x1 = (b.getX1() - lb.dx) / lb.scale;
			float y1 = (b.getY1() - lb.dy) / lb.scale;
			float x2 = (b.getX2() - lb.dx) / lb.scale;
			float y2 = (b.getY2() - lb.dy) / lb.scale;
			float[] lmk = null;
			float[] src = b.getLandmarks();
			if (src != null && src.length == 10) {
				lmk = new float[10];
				for (int i = 0; i < 10; i += 2) {
					lmk[i] = (src[i] - lb.dx) / lb.scale;
					lmk[i + 1] = (src[i + 1] - lb.dy) / lb.scale;
				}
			}
			out.add(new FaceBox(x1, y1, x2, y2, b.getScore(), lmk));
		}
		return out;
	}

	/**
	 * 根据 5 关键点（眼睛、鼻尖、嘴角）做仿射对齐，把人脸摆正到 {@code size x size}。
	 *
	 * <p>参考点是 ArcFace / SFace / InsightFace 等主流识别模型共享的"标准脸"模板。
	 * 没有关键点时退化为方形裁剪 + resize。
	 *
	 * @param src  原图
	 * @param box  检测框（含 landmarks，长度 10）
	 * @param size 目标边长（SFace / ArcFace 均为 112）
	 * @return 对齐后的 BGR 图像
	 */
	public static BufferedImage align(BufferedImage src, FaceBox box, int size) {
		float[] lmk = box.getLandmarks();
		if (lmk == null || lmk.length != 10) {
			return cropCenter(src, box, size);
		}
		// 取 左眼(0,1) 右眼(2,3) 鼻尖(4,5) 三个点求仿射变换
		float[] srcPts = new float[6];
		float[] dstPts = new float[6];
		System.arraycopy(lmk, 0, srcPts, 0, 6);
		dstPts[0] = ARC_FACE_REFERENCE[0][0];
		dstPts[1] = ARC_FACE_REFERENCE[0][1];
		dstPts[2] = ARC_FACE_REFERENCE[1][0];
		dstPts[3] = ARC_FACE_REFERENCE[1][1];
		dstPts[4] = ARC_FACE_REFERENCE[2][0];
		dstPts[5] = ARC_FACE_REFERENCE[2][1];
		double[] matrix = solveAffine(srcPts, dstPts);

		AffineTransform at = new AffineTransform(matrix);
		BufferedImage aligned = new BufferedImage(size, size, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = aligned.createGraphics();
		try {
			g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
			g.setColor(java.awt.Color.BLACK);
			g.fillRect(0, 0, size, size);
			g.setTransform(at);
			g.drawImage(src, 0, 0, null);
		} finally {
			g.dispose();
		}
		return aligned;
	}

	/**
	 * ArcFace / SFace 标准 5 关键点参考位置（112x112 坐标系下），与模型训练时保持一致。
	 */
	private static final float[][] ARC_FACE_REFERENCE = {
		{38.2946f, 51.6963f},   // 左眼
		{73.5318f, 51.5014f},   // 右眼
		{56.0252f, 71.7366f},   // 鼻尖
		{41.5493f, 92.3655f},   // 左嘴角
		{70.7299f, 92.2041f},   // 右嘴角
	};

	private static BufferedImage cropCenter(BufferedImage src, FaceBox box, int size) {
		int x = Math.max(0, (int) box.getX1());
		int y = Math.max(0, (int) box.getY1());
		int w = Math.min(src.getWidth() - x, (int) box.width());
		int h = Math.min(src.getHeight() - y, (int) box.height());
		if (w <= 0 || h <= 0) {
			throw new MicaAiException("Invalid face box: " + box);
		}
		BufferedImage crop = src.getSubimage(x, y, w, h);
		BufferedImage resized = new BufferedImage(size, size, BufferedImage.TYPE_3BYTE_BGR);
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
	 *
	 * <p>三个对应点足以唯一确定 6 参数的仿射变换。
	 * 这里采用 Cramer 法则对 u/v 分别求解 3x3 线性方程组：
	 * <pre>
	 *   [x0 y0 1] [a]   [u0]
	 *   [x1 y1 1] [b] = [u1]
	 *   [x2 y2 1] [c]   [u2]
	 * </pre>
	 * d/e/f 同理把 u 换成 v。
	 */
	private static double[] solveAffine(float[] src, float[] dst) {
		double x0 = src[0], y0 = src[1];
		double x1 = src[2], y1 = src[3];
		double x2 = src[4], y2 = src[5];
		double u0 = dst[0], v0 = dst[1];
		double u1 = dst[2], v1 = dst[3];
		double u2 = dst[4], v2 = dst[5];

		double det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1);
		if (Math.abs(det) < 1e-9) {
			return new double[]{1, 0, 0, 0, 1, 0};
		}
		double a = (u0 * (y1 - y2) - y0 * (u1 - u2) + (u1 * y2 - u2 * y1)) / det;
		double b = (x0 * (u1 - u2) - u0 * (x1 - x2) + (x1 * u2 - x2 * u1)) / det;
		double c = (x0 * (y1 * u2 - y2 * u1) - y0 * (x1 * u2 - x2 * u1) + u0 * (x1 * y2 - x2 * y1)) / det;
		double d = (v0 * (y1 - y2) - y0 * (v1 - v2) + (v1 * y2 - v2 * y1)) / det;
		double e = (x0 * (v1 - v2) - v0 * (x1 - x2) + (x1 * v2 - x2 * v1)) / det;
		double f = (x0 * (y1 * v2 - y2 * v1) - y0 * (x1 * v2 - x2 * v1) + v0 * (x1 * y2 - x2 * y1)) / det;
		return new double[]{a, b, c, d, e, f};
	}

	/**
	 * Letterbox 处理结果。
	 */
	public record LetterboxResult(BufferedImage image, float scale, int dx, int dy, int newW, int newH) {
	}
}
