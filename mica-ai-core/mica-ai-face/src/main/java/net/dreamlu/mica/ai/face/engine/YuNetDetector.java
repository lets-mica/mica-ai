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
import net.dreamlu.mica.ai.face.utils.ImageUtils;
import net.dreamlu.mica.ai.face.utils.ImageUtils.LetterboxResult;

import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

/**
 * YuNet 人脸检测引擎（OpenCV Zoo，纯 ONNX Runtime）。
 *
 * <p>对应模型：{@code face_detection_yunet_2023mar.onnx}，
 * 默认输入尺寸 320x320，模型内部已做 NMS，可直接用 score 阈值过滤。
 *
 * <p>模型输出：(1, 1, N, 15)，每个检测 = {@code [x, y, w, h, lmk_x1..lmk_y5, score]}。
 *
 * <p>License：Apache-2.0，可商用。
 *
 * @since 1.0.0
 */
@Slf4j
public class YuNetDetector implements FaceDetector {

	private final OrtSession session;
	private final FaceConfig config;

	public YuNetDetector(FaceConfig config) {
		this.config = config;
		try {
			OrtEnvironment env = OrtEnvironment.getEnvironment();
			OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
			opts.setIntraOpNumThreads(config.getIntraOpNumThreads());
			opts.setInterOpNumThreads(config.getInterOpNumThreads());
			this.session = env.createSession(config.getDetModelPath().toString(), opts);
			log.info("[mica-ai-face] YuNet loaded: {}", config.getDetModelPath());
		} catch (OrtException e) {
			throw new MicaAiException("Failed to load YuNet model: " + config.getDetModelPath(), e);
		}
	}

	@Override
	public List<FaceBox> detect(BufferedImage image) {
		try {
			int size = config.getDetInputSize();
			// 1. Letterbox 缩放到目标尺寸（保持长宽比）
			LetterboxResult lb = ImageUtils.letterbox(image, size);

			// 2. YuNet 预处理：RGB float32，0~255 像素，不归一化
			float[] input = ImageUtils.toNchwFloat(lb.image(), false);

			OnnxTensor tensor = OnnxTensor.createTensor(
				OrtEnvironment.getEnvironment(),
				FloatBuffer.wrap(input),
				new long[]{1, 3, size, size}
			);
			List<FaceBox> boxes;
			try (OrtSession.Result result = session.run(java.util.Collections.singletonMap("input", tensor))) {
				boxes = decode(result);
			} finally {
				tensor.close();
			}
			// 3. 还原到原图坐标
			return ImageUtils.mapBoxesBackToOriginal(boxes, lb);
		} catch (OrtException e) {
			throw new MicaAiException("YuNet inference failed", e);
		}
	}

	/**
	 * 解析 YuNet 输出 (1, 1, N, 15)。
	 */
	private List<FaceBox> decode(OrtSession.Result result) throws OrtException {
		float[] flat = null;
		for (Map.Entry<String, ai.onnxruntime.OnnxValue> e : result) {
			if (!(e.getValue() instanceof OnnxTensor t)) {
				continue;
			}
			Object v = t.getValue();
			if (v instanceof float[][][][] arr4d) {
				flat = arr4d[0][0][0];
				break;
			} else if (v instanceof float[][][] arr3d) {
				flat = arr3d[0][0];
				break;
			}
		}
		if (flat == null) {
			return new ArrayList<>();
		}
		// 每个检测 15 维：[x, y, w, h, lx1, ly1, ..., lx5, ly5, score]
		int numDetections = flat.length / 15;
		float scoreTh = config.getDetScoreThreshold();
		List<FaceBox> out = new ArrayList<>(numDetections);
		for (int i = 0; i < numDetections; i++) {
			int base = i * 15;
			float score = flat[base + 14];
			if (score < scoreTh) {
				continue;
			}
			float x = flat[base + 0];
			float y = flat[base + 1];
			float w = flat[base + 2];
			float h = flat[base + 3];
			float x1 = x;
			float y1 = y;
			float x2 = x + w;
			float y2 = y + h;
			float[] lmk = new float[10];
			System.arraycopy(flat, base + 4, lmk, 0, 10);
			out.add(new FaceBox(x1, y1, x2, y2, score, lmk));
		}
		out.sort(Comparator.comparingDouble(FaceBox::getScore));
		return out;
	}

	@Override
	public void close() {
		try {
			session.close();
		} catch (OrtException e) {
			log.warn("[mica-ai-face] failed to close YuNet session", e);
		}
	}
}
