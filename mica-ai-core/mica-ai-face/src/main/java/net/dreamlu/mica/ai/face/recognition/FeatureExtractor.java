/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.recognition;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * 人脸特征提取器，基于 <b>SFace</b> ONNX 模型
 * （OpenCV Zoo {@code face_recognition_sface_2021dec.onnx}）。
 *
 * <p>与 OpenCV {@code cv::FaceRecognizerSF} 的实现保持一致：
 * <ul>
 *     <li><b>输入</b>：{@code [1, 3, 112, 112]} float32，<b>RGB 顺序、原始 0~255 像素值</b>。
 *         OpenCV 用 {@code blobFromImage(img, 1, (112,112), Scalar(0,0,0), swapRB=true)} 构建输入。</li>
 *     <li><b>归一化内嵌在模型内</b>，外部<b>不要</b>再归一化。</li>
 *     <li><b>输出</b>：{@code [1, 128]}，未归一化的 128 维特征，本类负责 L2 归一化。</li>
 * </ul>
 */
public class FeatureExtractor {

	public static final int FEATURE_DIM = 128;
	private static final float SCALE = 1.0f;
	private static final float OFFSET = 0.0f;

	private final OrtSession session;
	private final OrtEnvironment env;

	public FeatureExtractor(ModelManager modelManager) {
		this.session = modelManager.getRecognitionSession();
		this.env = modelManager.getEnvironment();
	}

	public static float compare(float[] a, float[] b) {
		if (a == null || b == null) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.EXTRACTION_FAILED, "特征向量为空");
		}
		if (a.length == 0 || a.length != b.length) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.EXTRACTION_FAILED,
				"特征维度不匹配: " + a.length + "/" + b.length);
		}
		double dot = 0d;
		for (int i = 0; i < a.length; i++) {
			dot += (double) a[i] * b[i];
		}
		if (dot > 1d) {
			dot = 1d;
		} else if (dot < -1d) {
			dot = -1d;
		}
		return (float) dot;
	}

	private static void l2Normalize(float[] v) {
		double sum = 0d;
		for (float f : v) {
			sum += (double) f * f;
		}
		double norm = Math.sqrt(sum);
		if (norm < 1e-10) {
			return;
		}
		for (int i = 0; i < v.length; i++) {
			v[i] = (float) (v[i] / norm);
		}
	}

	public static float[] l2NormalizeCopy(float[] v) {
		float[] copy = v.clone();
		l2Normalize(copy);
		return copy;
	}

	public float[] extract(Mat alignedFace) {
		if (alignedFace == null || alignedFace.empty()) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.EXTRACTION_FAILED, "对齐人脸为空");
		}
		int h = alignedFace.rows();
		int w = alignedFace.cols();
		float[] data = ImageUtils.bgrHwcToRgbChwFloat(alignedFace, SCALE, OFFSET);

		try (OnnxTensor inputTensor = createInput(data, h, w)) {
			String inputName = session.getInputNames().iterator().next();
			Map<String, OnnxTensor> inputs = new HashMap<>();
			inputs.put(inputName, inputTensor);
			try (OrtSession.Result result = session.run(inputs)) {
				Object value = result.get(0).getValue();
				float[] feature = readFeature(value);
				l2Normalize(feature);
				return feature;
			} catch (OrtException e) {
				throw new MicaAiException(
					MicaAiException.ErrorCode.EXTRACTION_FAILED, "SFace 推理失败", e);
			}
		}
	}

	private OnnxTensor createInput(float[] data, int h, int w) {
		try {
			return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), new long[]{1, 3, h, w});
		} catch (OrtException e) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.EXTRACTION_FAILED, "构造 SFace 输入张量失败", e);
		}
	}

	private float[] readFeature(Object value) {
		if (value instanceof float[][]) {
			return ((float[][]) value)[0].clone();
		}
		if (value instanceof float[]) {
			return ((float[]) value).clone();
		}
		throw new MicaAiException(
			MicaAiException.ErrorCode.EXTRACTION_FAILED,
			"不支持的 SFace 输出类型: " + (value == null ? "null" : value.getClass().getName()));
	}
}