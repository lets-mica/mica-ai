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
package net.dreamlu.mica.ai.face.recognition;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
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
 *
 * <p>线程安全：ONNX Runtime OrtSession 线程安全，本类 stateless，可作为 Spring 单例 Bean 共享。
 */
public class FeatureExtractor {

	/**
	 * 输出特征维度：固定 128d，已 L2 归一化（点积 = 余弦相似度）。
	 */
	public static final int FEATURE_DIM = 128;
	private static final float SCALE = 1.0f;
	private static final float OFFSET = 0.0f;

	private final OrtSession session;
	private final OrtEnvironment env;

	/**
	 * 使用模型管理器构造特征提取器。
	 *
	 * @param modelManager 模型管理器，提供识别会话与环境
	 */
	public FeatureExtractor(ModelManager modelManager) {
		this.session = modelManager.getRecognitionSession();
		this.env = modelManager.getEnvironment();
	}

	/**
	 * 计算两 L2-归一化特征向量的余弦相似度（点积等价形式）。
	 *
	 * @param a 向量 a（非 null，长度 = {@link #FEATURE_DIM}）
	 * @param b 向量 b（非 null，长度需与 a 一致）
	 * @return 范围 {@code [-1, 1]} 的余弦相似度
	 * @throws MicaAiException {@link ErrorCode#EXTRACTION_FAILED} 向量为空或维度不匹配
	 */
	public static float compare(float[] a, float[] b) {
		if (a == null || b == null) {
			throw new MicaAiException(
				ErrorCode.EXTRACTION_FAILED, "特征向量为空");
		}
		if (a.length == 0 || a.length != b.length) {
			throw new MicaAiException(
				ErrorCode.EXTRACTION_FAILED,
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

	/**
	 * 对特征向量副本做 L2 归一化，不修改原数组。
	 *
	 * @param v 特征向量
	 * @return 归一化后的新数组
	 */
	public static float[] l2NormalizeCopy(float[] v) {
		float[] copy = v.clone();
		l2Normalize(copy);
		return copy;
	}

	/**
	 * 从 112×112 已对齐 BGR 人脸图提取 128d L2-归一化特征。
	 *
	 * @param alignedFace 由 {@code FaceAligner} 对齐后的 BGR Mat（112×112）
	 * @return 长度为 {@link #FEATURE_DIM} 的 float 数组，||v|| ≈ 1
	 * @throws MicaAiException {@link ErrorCode#EXTRACTION_FAILED} 入参为空 / 推理失败
	 */
	public float[] extract(Mat alignedFace) {
		if (alignedFace == null || alignedFace.empty()) {
			throw new MicaAiException(
				ErrorCode.EXTRACTION_FAILED, "对齐人脸为空");
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
					ErrorCode.EXTRACTION_FAILED, "SFace 推理失败", e);
			}
		}
	}

	private OnnxTensor createInput(float[] data, int h, int w) {
		try {
			return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), new long[]{1, 3, h, w});
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.EXTRACTION_FAILED, "构造 SFace 输入张量失败", e);
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
			ErrorCode.EXTRACTION_FAILED,
			"不支持的 SFace 输出类型: " + (value == null ? "null" : value.getClass().getName()));
	}
}