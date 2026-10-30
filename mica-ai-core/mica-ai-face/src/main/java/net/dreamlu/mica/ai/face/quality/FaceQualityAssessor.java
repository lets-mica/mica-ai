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
package net.dreamlu.mica.ai.face.quality;

import ai.onnxruntime.*;
import ai.onnxruntime.OrtSession.Result;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.face.config.FaceQualityConfig;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * eDifFIQA(T) 人脸质量评估门面。
 *
 * <p>吃对齐后的 112×112 BGR Mat（直接复用 {@code FaceAligner.align} 输出），
 * 推理输出一个 float32 quality score（越高越好；LFW 普遍 > 0.5，XQLFW 显著更低）。
 *
 * <pre>{@code
 * try (FaceQualityAssessor assessor = FaceQualityAssessor.create(
 *     FaceQualityConfig.builder().threshold(0.5f).build())) {
 *     Mat aligned = faceAligner.align(image, faceBox);
 *     try {
 *         FaceQualityResult r = assessor.assess(aligned);
 *         boolean ok = r.passed(assessor.getConfig().getThreshold());
 *     } finally {
 *         aligned.release();
 *     }
 * }
 * }</pre>
 *
 * <p>输入 / 输出节点名动态解析（不硬编码），换 eDifFIQA 同家族模型（eDifFIQA-S /
 * eDifFIQA-T / eDifFIQA-B）无需改 Java 代码。
 *
 * <p>线程安全：内部 {@link OrtSession} 线程安全；本类仅含构造时冻结的不可变字段，
 * 可作为 Spring 单例 Bean 共享。
 *
 * <p>License：CC-BY-4.0（署名即可商用，详见 model-tools/face/README.md）。
 */
@Slf4j
public class FaceQualityAssessor implements AutoCloseable {

	private static final int IMAGE_RANK = 4;

	@Getter
	private final FaceQualityConfig config;
	private final OnnxModelSession model;
	private final OrtEnvironment environment;
	private final String imageInputName;
	private final String scoreOutputName;

	public FaceQualityAssessor(FaceQualityConfig config) {
		if (config == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "FaceQualityConfig must not be null");
		}
		config.validate();
		this.config = config;
		this.environment = OrtEnvironment.getEnvironment();
		this.model = new OnnxModelSession(environment, config.resolveModelPath(),
			OrtSessionFactory.build(config.getOnnx()), "face-quality");
		this.imageInputName = resolveInputName();
		this.scoreOutputName = resolveScoreOutputName();
		validateImageInputSize();
		log.info("mica-ai-face quality 初始化完成: input={}, output={}, threshold={}, inputSize={}",
			imageInputName, scoreOutputName, config.getThreshold(), config.getInputSize());
	}

	public static FaceQualityAssessor create(FaceQualityConfig config) {
		return new FaceQualityAssessor(config);
	}

	public FaceQualityResult assess(Mat alignedFace) {
		if (alignedFace == null || alignedFace.empty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "alignedFace 为空");
		}
		float score;
		OnnxTensor inputTensor = null;
		try {
			float[] tensorData = preprocess(alignedFace);
			long[] shape = {1, 3, (long) config.getInputSize(), (long) config.getInputSize()};
			inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(tensorData), shape);
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(imageInputName, inputTensor);
			try (Result result = model.getSession().run(inputs)) {
				Object value = result.get(scoreOutputName).get().getValue();
				score = extractScalar(value);
			} catch (OrtException e) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "eDifFIQA 推理失败", e);
			}
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "创建 eDifFIQA 输入张量失败", e);
		} finally {
			closeQuietly(inputTensor);
		}
		return new FaceQualityResult(score);
	}

	private float[] preprocess(Mat alignedBgr) {
		Mat resized = new Mat();
		try {
			Imgproc.resize(alignedBgr, resized, new Size(config.getInputSize(), config.getInputSize()));
			return ImageUtils.bgrHwcToRgbChwFloat(resized, config.getMean(), config.getStd());
		} finally {
			resized.release();
		}
	}

	private String resolveInputName() {
		Map<String, String> byName = new LinkedHashMap<>();
		try {
			for (String name : model.getSession().getInputNames()) {
				byName.put(name.toLowerCase(Locale.ROOT), name);
			}
		} catch (RuntimeException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "读取 eDifFIQA 输入名失败", e);
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"eDifFIQA 模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		return byName.values().iterator().next();
	}

	private String resolveScoreOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = model.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "读取 eDifFIQA 输出元信息失败", e);
		}
		if (outputInfo.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"eDifFIQA 模型应有且仅有 1 个输出，实际 " + outputInfo.size() + " 个: " + outputInfo.keySet());
		}
		return outputInfo.keySet().iterator().next();
	}

	private void validateImageInputSize() {
		try {
			NodeInfo nodeInfo = model.getSession().getInputInfo().get(imageInputName);
			TensorInfo tensor = asTensorInfo(nodeInfo);
			if (tensor == null) {
				return;
			}
			long[] shape = tensor.getShape();
			if (shape.length != IMAGE_RANK) {
				throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
					"eDifFIQA 输入应为 4 维，实际 rank=" + shape.length);
			}
			long modelH = shape[IMAGE_RANK - 2];
			long modelW = shape[IMAGE_RANK - 1];
			if (modelH > 0 && modelW > 0
				&& (modelH != config.getInputSize() || modelW != config.getInputSize())) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"inputSize 必须等于模型输入边长 " + modelH + "，当前为 " + config.getInputSize());
			}
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "读取 eDifFIQA 模型输入形状失败", e);
		}
	}

	private static TensorInfo asTensorInfo(NodeInfo nodeInfo) {
		if (nodeInfo == null) {
			return null;
		}
		ValueInfo info = nodeInfo.getInfo();
		return info instanceof TensorInfo ? (TensorInfo) info : null;
	}

	private static float extractScalar(Object value) {
		if (value instanceof Float) {
			return (Float) value;
		}
		if (value instanceof float[] && ((float[]) value).length > 0) {
			return ((float[]) value)[0];
		}
		if (value instanceof float[][] && ((float[][]) value).length > 0
			&& ((float[][]) value)[0].length > 0) {
			return ((float[][]) value)[0][0];
		}
		if (value instanceof float[][][]) {
			float[][][] v = (float[][][]) value;
			if (v.length > 0 && v[0].length > 0 && v[0][0].length > 0) {
				return v[0][0][0];
			}
		}
		throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
			"eDifFIQA 输出无法解码为 float scalar: " + value.getClass());
	}

	private static void closeQuietly(OnnxTensor tensor) {
		if (tensor == null) {
			return;
		}
		try {
			tensor.close();
		} catch (Exception ignore) {
		}
	}

	@Override
	public void close() {
		try {
			model.close();
		} catch (Exception e) {
			log.warn("关闭 FaceQualityAssessor 的 ONNX session 失败: {}", e.getMessage());
		}
	}
}
