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
package net.dreamlu.mica.ai.face.expression;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.face.config.ExpressionConfig;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Progressive Teacher 表情识别门面（MobileFaceNet backbone，7 类 softmax）。
 *
 * <p>吃对齐后的 112×112 BGR Mat（直接复用 {@code FaceAligner.align} 输出），
 * 推理输出 7 维 softmax 概率（angry / disgust / fearful / happy / neutral / sad / surprised）。
 *
 * <pre>{@code
 * try (ExpressionRecognizer rec = ExpressionRecognizer.create(ExpressionConfig.builder().build())) {
 *     Mat aligned = faceAligner.align(image, faceBox);
 *     try {
 *         ExpressionResult r = rec.recognize(aligned);
 *         System.out.println(r.getExpression() + " " + r.getScore());
 *     } finally {
 *         aligned.release();
 *     }
 * }
 * }</pre>
 *
 * <p>输入节点名动态解析；输出节点名默认为 {@code label}（opencv_zoo 官方导出形态），
 * 若实测不符会 fallback 到第一个 float 输出。
 *
 * <p>线程安全：内部 {@link OrtSession} 线程安全；本类仅含构造时冻结的不可变字段，
 * 可作为 Spring 单例 Bean 共享。
 *
 * <p>License：Apache-2.0（opencv_zoo 目录声明）。
 */
@Slf4j
public class ExpressionRecognizer implements AutoCloseable {

	private static final int IMAGE_RANK = 4;
	private static final int NUM_CLASSES = 7;
	private static final String PREFERRED_OUTPUT = "label";

	@Getter
	private final ExpressionConfig config;
	private final OnnxModelSession model;
	private final OrtEnvironment environment;
	private final String imageInputName;
	private final String labelOutputName;

	public ExpressionRecognizer(ExpressionConfig config) {
		if (config == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "ExpressionConfig must not be null");
		}
		config.validate();
		this.config = config;
		this.environment = OrtEnvironment.getEnvironment();
		this.model = new OnnxModelSession(environment, config.resolveModelPath(),
			OrtSessionFactory.build(config.getOnnx()), "face-expression");
		this.imageInputName = resolveInputName();
		this.labelOutputName = resolveLabelOutputName();
		validateImageInputSize();
		log.info("mica-ai-face expression 初始化完成: input={}, output={}, inputSize={}",
			imageInputName, labelOutputName, config.getInputSize());
	}

	public static ExpressionRecognizer create(ExpressionConfig config) {
		return new ExpressionRecognizer(config);
	}

	public ExpressionResult recognize(Mat alignedFace) {
		if (alignedFace == null || alignedFace.empty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "alignedFace 为空");
		}
		OnnxTensor inputTensor = null;
		try {
			float[] tensorData = preprocess(alignedFace);
			long[] shape = {1, 3, (long) config.getInputSize(), (long) config.getInputSize()};
			inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(tensorData), shape);
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(imageInputName, inputTensor);
			float[] probs;
			try (Result result = model.getSession().run(inputs)) {
				Object value = result.get(labelOutputName).get().getValue();
				float[] raw = extractVector(value);
				if (raw.length != NUM_CLASSES) {
					log.warn("表情识别输出维度 {} != 7，跳过 softmax", raw.length);
					probs = raw;
				} else {
					probs = softmax(raw);
				}
			} catch (OrtException e) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "表情识别推理失败", e);
			}
			int argmax = argmax(probs);
			return new ExpressionResult(ExpressionType.fromIndex(argmax), probs[argmax], probs);
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "创建表情识别输入张量失败", e);
		} finally {
			closeQuietly(inputTensor);
		}
	}

	private float[] preprocess(Mat alignedBgr) {
		Mat rgb = new Mat();
		Mat resized = new Mat();
		Mat floatMat = new Mat();
		try {
			Imgproc.cvtColor(alignedBgr, rgb, Imgproc.COLOR_BGR2RGB);
			Imgproc.resize(rgb, resized, new Size(config.getInputSize(), config.getInputSize()));
			resized.convertTo(floatMat, CvType.CV_32FC3, 1.0 / 255.0);
			float[] mean = config.getMean();
			float[] std = config.getStd();
			int total = floatMat.rows() * floatMat.cols() * 3;
			float[] data = new float[total];
			int idx = 0;
			for (int y = 0; y < floatMat.rows(); y++) {
				for (int x = 0; x < floatMat.cols(); x++) {
					double[] pixel = floatMat.get(y, x);
					for (int c = 0; c < 3; c++) {
						data[idx++] = (float) ((pixel[c] - mean[c]) / std[c]);
					}
				}
			}
			return data;
		} finally {
			rgb.release();
			resized.release();
			floatMat.release();
		}
	}

	private String resolveInputName() {
		Map<String, String> byName = new LinkedHashMap<>();
		try {
			for (String name : model.getSession().getInputNames()) {
				byName.put(name.toLowerCase(Locale.ROOT), name);
			}
		} catch (RuntimeException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "读取表情识别输入名失败", e);
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"表情识别模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		return byName.values().iterator().next();
	}

	private String resolveLabelOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = model.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "读取表情识别输出元信息失败", e);
		}
		for (String name : outputInfo.keySet()) {
			if (PREFERRED_OUTPUT.equals(name)) {
				return name;
			}
		}
		if (outputInfo.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"表情识别模型应有且仅有 1 个输出，实际 " + outputInfo.size() + " 个: " + outputInfo.keySet());
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
					"表情识别输入应为 4 维，实际 rank=" + shape.length);
			}
			long modelH = shape[IMAGE_RANK - 2];
			long modelW = shape[IMAGE_RANK - 1];
			if (modelH > 0 && modelW > 0
				&& (modelH != config.getInputSize() || modelW != config.getInputSize())) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"inputSize 必须等于模型输入边长 " + modelH + "，当前为 " + config.getInputSize());
			}
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "读取表情识别模型输入形状失败", e);
		}
	}

	private static TensorInfo asTensorInfo(NodeInfo nodeInfo) {
		if (nodeInfo == null) {
			return null;
		}
		ValueInfo info = nodeInfo.getInfo();
		return info instanceof TensorInfo ? (TensorInfo) info : null;
	}

	private static float[] extractVector(Object value) {
		if (value instanceof float[]) {
			return (float[]) value;
		}
		if (value instanceof float[][]) {
			float[][] two = (float[][]) value;
			return two.length > 0 ? two[0] : new float[0];
		}
		if (value instanceof float[][][]) {
			float[][][] three = (float[][][]) value;
			return three.length > 0 && three[0].length > 0 ? three[0][0] : new float[0];
		}
		throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
			"表情识别输出无法解码为 float 向量: " + value.getClass());
	}

	private static float[] softmax(float[] logits) {
		float max = logits[0];
		for (int i = 1; i < logits.length; i++) {
			if (logits[i] > max) {
				max = logits[i];
			}
		}
		float[] exp = new float[logits.length];
		float sum = 0f;
		for (int i = 0; i < logits.length; i++) {
			exp[i] = (float) Math.exp(logits[i] - max);
			sum += exp[i];
		}
		for (int i = 0; i < logits.length; i++) {
			exp[i] /= sum;
		}
		return exp;
	}

	private static int argmax(float[] probs) {
		int idx = 0;
		for (int i = 1; i < probs.length; i++) {
			if (probs[i] > probs[idx]) {
				idx = i;
			}
		}
		return idx;
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
			log.warn("关闭 ExpressionRecognizer 的 ONNX session 失败: {}", e.getMessage());
		}
	}
}
