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
import net.dreamlu.mica.ai.face.config.FaceBox;
import net.dreamlu.mica.ai.face.config.FaceConfig;
import net.dreamlu.mica.ai.face.preprocess.ImageUtils;
import net.dreamlu.mica.ai.face.preprocess.ImageUtils.LetterboxResult;

import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * RetinaFace 人脸检测引擎（纯 ONNX Runtime）。
 *
 * <p>对应 buffalo_l 模型集中的 {@code det_10g.onnx}，固定输入 640x640。
 *
 * <p>模型输出（按 InsightFace 标准命名）：
 * <ul>
 *   <li>443：每个 anchor 的 4 个框回归 + 1 个分类 = 5 个值</li>
 *   <li>451：每个 anchor 的 10 个 landmark 偏移</li>
 *   <li>459：每个 anchor 的 kps（兼容命名）</li>
 * </ul>
 *
 * @since 1.0.0
 */
@Slf4j
public class RetinaFaceDetector implements Closeable, AutoCloseable {

	private final OrtSession session;
	private final FaceConfig config;

	public RetinaFaceDetector(FaceConfig config) {
		this.config = config;
		try {
			OrtEnvironment env = OrtEnvironment.getEnvironment();
			OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
			opts.setIntraOpNumThreads(config.getIntraOpNumThreads());
			opts.setInterOpNumThreads(config.getInterOpNumThreads());
			this.session = env.createSession(config.getDetModelPath().toString(), opts);
			log.info("[mica-ai-face] RetinaFace loaded: {}", config.getDetModelPath());
		} catch (OrtException e) {
			throw new MicaAiException("Failed to load RetinaFace model: " + config.getDetModelPath(), e);
		}
	}

	/**
	 * 检测图片中所有人脸。
	 *
	 * @return 人脸框列表（已按 score 降序）
	 */
	public List<FaceBox> detect(BufferedImage img) {
		try {
			LetterboxResult lb = ImageUtils.letterbox(img, config.getDetInputSize());
			float[] input = ImageUtils.toNchwFloat(lb.image());
			OnnxTensor tensor = OnnxTensor.createTensor(
				OrtEnvironment.getEnvironment(),
				FloatBuffer.wrap(input),
				new long[]{1, 3, config.getDetInputSize(), config.getDetInputSize()}
			);
			List<FaceBox> boxes;
			try (OrtSession.Result result = session.run(java.util.Collections.singletonMap("input.1", tensor))) {
				boxes = decode(result);
			} finally {
				tensor.close();
			}
			boxes = nonMaxSuppression(boxes, config.getDetNmsThreshold());
			// 还原到原图坐标
			return ImageUtils.mapBoxesBackToOriginal(boxes, lb);
		} catch (OrtException e) {
			throw new MicaAiException("RetinaFace inference failed", e);
		}
	}

	private List<FaceBox> decode(OrtSession.Result result) throws OrtException {
		// 三个输出，按名字特征模糊匹配
		float[][][] bbox = null;
		float[][][] landmark = null;
		for (java.util.Map.Entry<String, ai.onnxruntime.OnnxValue> e : result) {
			String name = e.getKey().toLowerCase();
			if (!(e.getValue() instanceof OnnxTensor t)) {
				continue;
			}
			float[] flat = ((float[][][]) t.getValue())[0][0];
			// scores 形状 [1, N, 1]
			if (flat.length == 0) {
				continue;
			}
			if (name.contains("448") || name.contains("bbox") || name.contains("cls") || name.contains("scr")) {
				if (flat.length / 5 == (int) (flat.length / 5.0)) {
					// 可能是 4 reg + 1 cls
					int n = flat.length / 5;
					if (bbox == null) {
						bbox = new float[n][1][5];
						for (int i = 0; i < n; i++) {
							System.arraycopy(flat, i * 5, bbox[i][0], 0, 5);
						}
					}
				}
			} else if (name.contains("451") || name.contains("landmark") || name.contains("kps")) {
				int n = flat.length / 10;
				landmark = new float[n][1][10];
				for (int i = 0; i < n; i++) {
					System.arraycopy(flat, i * 10, landmark[i][0], 0, 10);
				}
			}
		}
		List<FaceBox> out = new ArrayList<>();
		if (bbox == null) {
			return out;
		}
		float scoreTh = config.getDetScoreThreshold();
		for (int i = 0; i < bbox.length; i++) {
			float[] v = bbox[i][0];
			float score = sigmoid(v[4]);
			if (score < scoreTh) {
				continue;
			}
			float cx = v[0];
			float cy = v[1];
			float w = v[2];
			float h = v[3];
			float x1 = cx - w / 2f;
			float y1 = cy - h / 2f;
			float x2 = cx + w / 2f;
			float y2 = cy + h / 2f;
			float[] lmk = null;
			if (landmark != null && i < landmark.length) {
				lmk = landmark[i][0].clone();
			}
			out.add(new FaceBox(x1, y1, x2, y2, score, lmk));
		}
		out.sort(Comparator.comparingDouble(FaceBox::getScore));
		return out;
	}

	/**
	 * 简单 NMS：按 score 降序，依次加入保留集合，IoU 大于阈值的丢弃。
	 */
	private List<FaceBox> nonMaxSuppression(List<FaceBox> boxes, float iouThreshold) {
		List<FaceBox> sorted = new ArrayList<>(boxes);
		sorted.sort((a, b) -> Float.compare(b.getScore(), a.getScore()));
		List<FaceBox> keep = new ArrayList<>();
		boolean[] suppressed = new boolean[sorted.size()];
		for (int i = 0; i < sorted.size(); i++) {
			if (suppressed[i]) {
				continue;
			}
			FaceBox a = sorted.get(i);
			keep.add(a);
			for (int j = i + 1; j < sorted.size(); j++) {
				if (suppressed[j]) {
					continue;
				}
				if (iou(a, sorted.get(j)) > iouThreshold) {
					suppressed[j] = true;
				}
			}
		}
		return keep;
	}

	private float iou(FaceBox a, FaceBox b) {
		float xx1 = Math.max(a.getX1(), b.getX1());
		float yy1 = Math.max(a.getY1(), b.getY1());
		float xx2 = Math.min(a.getX2(), b.getX2());
		float yy2 = Math.min(a.getY2(), b.getY2());
		float w = Math.max(0f, xx2 - xx1);
		float h = Math.max(0f, yy2 - yy1);
		float inter = w * h;
		float union = a.area() + b.area() - inter;
		return union <= 0 ? 0 : inter / union;
	}

	private static float sigmoid(float x) {
		return (float) (1.0 / (1.0 + Math.exp(-x)));
	}

	@Override
	public void close() {
		try {
			session.close();
		} catch (OrtException e) {
			log.warn("[mica-ai-face] failed to close RetinaFace session", e);
		}
	}
}
