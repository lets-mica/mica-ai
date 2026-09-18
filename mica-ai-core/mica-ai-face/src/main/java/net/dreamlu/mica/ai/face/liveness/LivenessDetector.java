/*
 * Copyright (c) 2019-2029, Dreamlu 卢春梦 (596392912@qq.com & dreamlu.net).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.dreamlu.mica.ai.face.liveness;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
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
 * 活体检测器，基于 MiniFASNetV2 ONNX 模型（{@code 2.7_80x80_MiniFASNetV2.onnx}）。
 *
 * <p><b>类别顺序</b>：<code>0 = 纸质照片攻击(print)、1 = 真人(real)、2 = 屏幕翻拍攻击(replay)</code>，
 * 活体概率取 <b>下标 1</b>。
 *
 * <p><b>输入</b>：原图 + 人脸框，按 {@code cropScale}（默认 2.7）外扩裁剪到 80×80 BGR，
 * <b>不要</b>直接传对齐后的 112×112 人脸，否则视野太窄影响精度。
 *
 * <p>线程安全：依赖 {@code ModelManager} 持有的 ONNX Session（线程安全），可作为
 * Spring 单例 bean；本类仅含构造时冻结的不可变字段。
 */
@Getter
public class LivenessDetector {

	/** 模型输入边长（像素）。 */
	public static final int INPUT_SIZE = 80;
	/** softmax 输出中纸质照片攻击（print）的下标。 */
	public static final int INDEX_PRINT = 0;
	/** softmax 输出中真人（live）的下标。 */
	public static final int INDEX_LIVE = 1;
	/** softmax 输出中屏幕翻拍攻击（replay）的下标。 */
	public static final int INDEX_REPLAY = 2;

	private static final float SCALE = 1.0f;
	private static final float OFFSET = 0.0f;

	private final OrtSession session;
	private final OrtEnvironment env;
	private final float threshold;
	private final double cropScale;

	/**
	 * 构造活体检测器（默认阈值 0.85、外扩 2.7 倍）。
	 *
	 * @param modelManager 模型管理器（活体模型需已加载）
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 活体模型未加载
	 */
	public LivenessDetector(ModelManager modelManager) {
		this(modelManager, 0.85f, 2.7);
	}

	/**
	 * 构造活体检测器。
	 *
	 * @param modelManager 模型管理器（活体 session 必须存在）
	 * @param threshold    真人概率阈值；{@code > threshold} 判为 live
	 * @param cropScale    人脸框外扩倍数（推荐 2.7，与训练时一致）
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 活体模型未加载
	 */
	public LivenessDetector(ModelManager modelManager, float threshold, double cropScale) {
		OrtSession s = modelManager.getLivenessSession();
		if (s == null) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED,
				"活体模型未加载，无法创建 LivenessDetector");
		}
		this.session = s;
		this.env = modelManager.getEnvironment();
		this.threshold = threshold;
		this.cropScale = cropScale;
	}

	/**
	 * 对 logits 做数值稳定的 softmax 归一化。
	 *
	 * @param logits 模型输出的原始 logits
	 * @return 归一化后的概率分布（和为 1）
	 */
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

	/**
	 * 单帧活体检测。
	 *
	 * @param image 原图（BGR）
	 * @param box   人脸框（YuNet 检出）
	 * @return {@link LivenessResult}，含真人概率 + 是否通过 + 攻击类型
	 * @throws MicaAiException {@link ErrorCode#LIVENESS_FAILED} 入参为空 / 推理失败
	 */
	public LivenessResult check(Mat image, FaceBox box) {
		if (image == null || image.empty() || box == null) {
			throw new MicaAiException(
				ErrorCode.LIVENESS_FAILED, "活体检测入参为空");
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
						ErrorCode.LIVENESS_FAILED, "活体推理失败", e);
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
			ErrorCode.LIVENESS_FAILED,
			"不支持的活体模型输出类型: " + (value == null ? "null" : value.getClass().getName()));
	}

	private OnnxTensor createInput(float[] data) {
		try {
			return OnnxTensor.createTensor(env, FloatBuffer.wrap(data),
				new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.LIVENESS_FAILED, "构造活体输入张量失败", e);
		}
	}
}