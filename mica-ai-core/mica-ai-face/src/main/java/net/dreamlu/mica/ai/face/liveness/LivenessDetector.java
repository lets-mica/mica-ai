/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.liveness;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.LivenessResult;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * 活体检测器，基于 MiniFASNetV2 ONNX 模型。
 *
 * <p><b>类别顺序</b>：<code>0 = 纸质照片攻击(print)、1 = 真人(real)、2 = 屏幕翻拍攻击(replay)</code>，
 * 活体概率取 <b>下标 1</b>。
 */
@Getter
public class LivenessDetector {

	public static final int INPUT_SIZE = 80;
	public static final int INDEX_PRINT = 0;
	public static final int INDEX_LIVE = 1;
	public static final int INDEX_REPLAY = 2;

	private static final float SCALE = 1.0f;
	private static final float OFFSET = 0.0f;

	private final OrtSession session;
	private final OrtEnvironment env;
	private final float threshold;
	private final double cropScale;

	public LivenessDetector(ModelManager modelManager) {
		this(modelManager, 0.85f, 2.7);
	}

	public LivenessDetector(ModelManager modelManager, float threshold, double cropScale) {
		OrtSession s = modelManager.getLivenessSession();
		if (s == null) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.MODEL_LOAD_FAILED,
				"活体模型未加载，无法创建 LivenessDetector");
		}
		this.session = s;
		this.env = modelManager.getEnvironment();
		this.threshold = threshold;
		this.cropScale = cropScale;
	}

	public static float[] softmax(float[] logits) {
		float max = Float.NEGATIVE_INFINITY;
		for (float v : logits) {
			if (v > max) {
				max = v;
			}
		}
		float[] result = new float[logits.length];
		double sum = 0d;
		for (int i = 0; i < logits.length; i++) {
			result[i] = (float) Math.exp(logits[i] - max);
			sum += result[i];
		}
		if (sum < 1e-10) {
			Arrays.fill(result, 1f / result.length);
			return result;
		}
		for (int i = 0; i < result.length; i++) {
			result[i] = (float) (result[i] / sum);
		}
		return result;
	}

	public LivenessResult check(Mat image, FaceBox box) {
		if (image == null || image.empty() || box == null) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.LIVENESS_FAILED, "活体检测入参为空");
		}
		Mat cropped = ImageUtils.cropWithMargin(image, box, cropScale, INPUT_SIZE);
		try {
			float[] data = ImageUtils.bgrHwcToChwFloat(cropped, SCALE, OFFSET);
			try (OnnxTensor inputTensor = createInput(data)) {
				String inputName = session.getInputNames().iterator().next();
				Map<String, OnnxTensor> inputs = new HashMap<>();
				inputs.put(inputName, inputTensor);
				try (OrtSession.Result result = session.run(inputs)) {
					float[] logits = readLogits(result.get(0).getValue());
					float[] probs = softmax(logits);

					float liveScore = probs[INDEX_LIVE];
					boolean isLive = liveScore > threshold;
					String attackType;
					if (isLive) {
						attackType = "real";
					} else if (probs.length > INDEX_REPLAY) {
						attackType = probs[INDEX_PRINT] >= probs[INDEX_REPLAY] ? "print" : "replay";
					} else {
						attackType = "unknown";
					}
					return new LivenessResult(liveScore, isLive, attackType);
				} catch (OrtException e) {
					throw new MicaAiException(
						MicaAiException.ErrorCode.LIVENESS_FAILED, "活体推理失败", e);
				}
			}
		} finally {
			cropped.release();
		}
	}

	private float[] readLogits(Object value) {
		if (value instanceof float[][]) {
			return ((float[][]) value)[0];
		}
		if (value instanceof float[]) {
			return ((float[]) value);
		}
		throw new MicaAiException(
			MicaAiException.ErrorCode.LIVENESS_FAILED,
			"不支持的活体模型输出类型: " + (value == null ? "null" : value.getClass().getName()));
	}

	private OnnxTensor createInput(float[] data) {
		try {
			return OnnxTensor.createTensor(env, FloatBuffer.wrap(data),
				new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
		} catch (OrtException e) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.LIVENESS_FAILED, "构造活体输入张量失败", e);
		}
	}
}