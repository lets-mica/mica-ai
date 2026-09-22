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
package net.dreamlu.mica.ai.textline.detection;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.textline.config.TextLineConfig;
import net.dreamlu.mica.ai.textline.model.TextLineOrientation;
import net.dreamlu.mica.ai.textline.model.TextLineOrientationResult;
import net.dreamlu.mica.ai.textline.util.TextLineImageUtils;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * PP-LCNet 文本行方向分类器（官方 ONNX 导出形态，I/O 实测于 2026-09-21）。
 *
 * <p>ONNX 输入：
 * <ul>
 *   <li>{@code x}：{@code [N, 3, 80, 160]} float32（动态 batch），
 *       缩放 + CHW + {@code (x/255-mean)/std}</li>
 * </ul>
 * ONNX 输出：
 * <ul>
 *   <li>{@code fetch_name_0}：{@code [N, 2]} float32，<b>原始 logits</b>（未 softmax），
 *       索引 0 = {@code 0_degree}、索引 1 = {@code 180_degree}</li>
 * </ul>
 *
 * <p>类别顺序来自官方推理包 {@code inference.yml} 的
 * {@code PostProcess.Topk.label_list: [0_degree, 180_degree]}，不可按字典序猜测。
 *
 * <p>输入 / 输出节点名均按名称提示解析（{@code x} / {@code input} / {@code images}，
 * 含子串兜底），保证换导出不会静默错位；输出名 {@code fetch_name_0} 是 Paddle2ONNX
 * 的自动命名，同样按「2 类 float 输出」的结构兜底。
 *
 * <p>线程安全：ONNX session 线程安全，每次推理新建 1 个 {@code OnnxTensor}；
 * 通常由 {@link net.dreamlu.mica.ai.textline.TextLineEngine} 单例持有。
 */
@Slf4j
public class TextLineDetector implements AutoCloseable {

	private static final int IMAGE_RANK = 4;
	private static final int CHANNEL_DIM = 1;
	private static final int MAX_CLASSES = 100;

	private final OrtEnvironment environment;
	private final OnnxModelSession session;
	@Getter
	private final TextLineConfig config;
	private final String imageInputName;
	private final String logitsOutputName;

	/**
	 * 构造文本行方向分类器，加载 ONNX 模型并解析输入输出张量名。
	 *
	 * @param environment    ONNX 运行环境
	 * @param config         方向分类配置
	 * @param sessionOptions ONNX 会话选项
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载或张量名解析失败
	 */
	public TextLineDetector(OrtEnvironment environment, TextLineConfig config,
							OrtSession.SessionOptions sessionOptions) {
		this.environment = environment;
		this.config = config;
		this.session = new OnnxModelSession(
			environment, config.resolveModelPath(),
			sessionOptions, "textline");
		this.imageInputName = resolveInputName();
		this.logitsOutputName = resolveOutputName();
		validateImageInputShape();
		log.info("mica-ai-textline 初始化完成: version={} input={} output={} order={}",
			config.getModelVersion(), imageInputName, logitsOutputName, config.getChannelOrder());
	}

	/**
	 * 判定文本行方向。
	 *
	 * @param bgr 文本行图（BGR）；null 或空图时返回 null
	 * @return 方向判定结果；输入为空时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 预处理或 ONNX 推理失败
	 */
	public TextLineOrientationResult classify(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return null;
		}
		float[] tensorData = TextLineImageUtils.resizeToChwFloat(
			bgr, config.getInputWidth(), config.getInputHeight(),
			config.getMean(), config.getStd(),
			config.getChannelOrder(), config.getInterpolation());

		OnnxTensor inputTensor = null;
		try {
			inputTensor = OnnxTensor.createTensor(environment,
				FloatBuffer.wrap(tensorData),
				new long[]{1, 3, config.getInputHeight(), config.getInputWidth()});
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(imageInputName, inputTensor);
			try (OrtSession.Result result = session.getSession().run(inputs)) {
				return extractOrientation(result);
			} catch (OrtException e) {
				throw new MicaAiException(
					ErrorCode.INFERENCE_FAILED, "textline 模型推理失败", e);
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "创建 textline 输入张量失败", e);
		} finally {
			closeQuietly(inputTensor);
		}
	}

	/**
	 * 解码字节数组为 BGR Mat 并跑 {@link #classify(Mat)}，自动 release 临时 Mat。
	 *
	 * @param imageBytes 图像字节（非空）
	 * @return 方向判定结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public TextLineOrientationResult classifyBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = TextLineImageUtils.byteArrayToMat(imageBytes);
			return classify(bgr);
		} finally {
			TextLineImageUtils.releaseAll(bgr);
		}
	}

	private TextLineOrientationResult extractOrientation(OrtSession.Result result) throws OrtException {
		Object value = result.get(logitsOutputName).get().getValue();
		float[] logits = asOneDim(value);
		if (logits == null || logits.length == 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"textline 输出 " + logitsOutputName + " 不是 1 维 float 张量");
		}
		if (logits.length != 2) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"textline 期望 2 类输出，实际为 " + logits.length
					+ " 类；请确认模型是否为文本行方向分类模型");
		}

		float[] probs = config.isOutputIsProbability() ? logits : softmax(logits);
		int argMax = probs[0] >= probs[1] ? 0 : 1;
		// 未达阈值时按「方向正常」处理，避免误旋转本来正确的行
		if (argMax == 1 && probs[1] < config.getUpsideDownThreshold()) {
			argMax = 0;
		}
		return new TextLineOrientationResult(
			TextLineOrientation.fromClassIndex(argMax), probs[argMax], logits);
	}

	private String resolveInputName() {
		Map<String, String> byName = new LinkedHashMap<>();
		try {
			for (String name : session.getSession().getInputNames()) {
				byName.put(name.toLowerCase(Locale.ROOT), name);
			}
		} catch (RuntimeException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 textline 模型输入名失败", e);
		}
		String name = pickByName(byName, "x", "input", "images", "image");
		if (name == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"textline 模型缺少输入节点（期望 x / input / images）");
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"textline 模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		return name;
	}

	/**
	 * 定位 2 类分类输出节点。
	 *
	 * <p>官方导出名为 {@code fetch_name_0}（Paddle2ONNX 自动命名），不可硬编码；
	 * 先按名称提示匹配，再回落到「唯一的 2 维 float 输出」结构判定。
	 */
	private String resolveOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = session.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 textline 模型输出元信息失败", e);
		}
		Map<String, String> byName = new LinkedHashMap<>();
		for (String name : outputInfo.keySet()) {
			byName.put(name.toLowerCase(Locale.ROOT), name);
		}
		String hinted = pickByName(byName, "fetch_name_0", "logits", "output", "prob", "softmax");
		if (hinted != null) {
			return hinted;
		}
		String matched = null;
		for (Map.Entry<String, NodeInfo> entry : outputInfo.entrySet()) {
			TensorInfo tensor = asTensorInfo(entry.getValue());
			if (tensor == null || tensor.type != OnnxJavaType.FLOAT) {
				continue;
			}
			long[] shape = tensor.getShape();
			if (shape.length != 2) {
				continue;
			}
			long classes = shape[shape.length - 1];
			if (classes <= 0 || classes > MAX_CLASSES) {
				continue;
			}
			if (matched == null) {
				matched = entry.getKey();
			}
		}
		if (matched == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"未能在 textline 模型中定位分类输出（期望 float [N, classes]）");
		}
		log.info("textline 输出节点按结构兜底解析: {}", matched);
		return matched;
	}

	/**
	 * 校验模型输入形状与配置一致，否则缩放结果与模型张量形状对不上。
	 */
	private void validateImageInputShape() {
		try {
			NodeInfo nodeInfo = session.getSession().getInputInfo().get(imageInputName);
			TensorInfo tensor = asTensorInfo(nodeInfo);
			if (tensor == null) {
				return;
			}
			long[] shape = tensor.getShape();
			if (shape.length != IMAGE_RANK) {
				throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
					"textline 模型 input 应为 4 维，实际 rank=" + shape.length);
			}
			long modelH = shape[IMAGE_RANK - 2];
			long modelW = shape[IMAGE_RANK - 1];
			if (modelH > 0 && modelH != config.getInputHeight()) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"inputHeight 必须等于模型输入高度 " + modelH + "，当前为 "
						+ config.getInputHeight());
			}
			if (modelW > 0 && modelW != config.getInputWidth()) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"inputWidth 必须等于模型输入宽度 " + modelW + "，当前为 "
						+ config.getInputWidth());
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 textline 模型输入形状失败", e);
		}
	}

	private static float[] softmax(float[] logits) {
		float max = logits[0];
		for (float v : logits) {
			if (v > max) {
				max = v;
			}
		}
		double sum = 0d;
		double[] exp = new double[logits.length];
		for (int i = 0; i < logits.length; i++) {
			exp[i] = Math.exp(logits[i] - max);
			sum += exp[i];
		}
		float[] out = new float[logits.length];
		for (int i = 0; i < logits.length; i++) {
			out[i] = (float) (exp[i] / sum);
		}
		return out;
	}

	private static String pickByName(Map<String, String> meta, String... hints) {
		for (String hint : hints) {
			String exact = meta.get(hint.toLowerCase(Locale.ROOT));
			if (exact != null) {
				return exact;
			}
		}
		for (String hint : hints) {
			for (Map.Entry<String, String> e : meta.entrySet()) {
				if (e.getKey().contains(hint.toLowerCase(Locale.ROOT))) {
					return e.getValue();
				}
			}
		}
		return null;
	}

	private static TensorInfo asTensorInfo(NodeInfo nodeInfo) {
		if (nodeInfo == null) {
			return null;
		}
		ValueInfo info = nodeInfo.getInfo();
		return info instanceof TensorInfo ? (TensorInfo) info : null;
	}

	private static float[] asOneDim(Object value) {
		if (value instanceof float[]) {
			return (float[]) value;
		}
		if (value instanceof float[][]) {
			float[][] two = (float[][]) value;
			return two.length > 0 ? two[0] : null;
		}
		if (value instanceof float[][][]) {
			float[][][] three = (float[][][]) value;
			return three.length > 0 && three[0].length > 0 ? three[0][0] : null;
		}
		return null;
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

	/**
	 * 释放底层 ONNX session。
	 */
	@Override
	public void close() {
		if (session != null) {
			session.close();
		}
	}

	/**
	 * 便于测试断言：模型输入宽度（NCHW 最后一维）。
	 *
	 * @return 输入宽度；读取失败返回 -1
	 */
	public int modelInputWidth() {
		return (int) inputShapeDim(IMAGE_RANK - 1);
	}

	/**
	 * 便于测试断言：模型输入高度（NCHW 倒数第二维）。
	 *
	 * @return 输入高度；读取失败返回 -1
	 */
	public int modelInputHeight() {
		return (int) inputShapeDim(IMAGE_RANK - 2);
	}

	/**
	 * 便于测试断言：模型分类数（输出最后一维）。
	 *
	 * @return 分类数；读取失败返回 -1
	 */
	public int classCount() {
		try {
			TensorInfo tensor = asTensorInfo(
				session.getSession().getOutputInfo().get(logitsOutputName));
			if (tensor == null) {
				return -1;
			}
			long[] shape = tensor.getShape();
			return shape.length == 2 && shape[1] > 0 ? (int) shape[1] : -1;
		} catch (OrtException e) {
			return -1;
		}
	}

	private long inputShapeDim(int index) {
		try {
			TensorInfo tensor = asTensorInfo(
				session.getSession().getInputInfo().get(imageInputName));
			if (tensor == null) {
				return -1L;
			}
			long[] shape = tensor.getShape();
			return shape.length == IMAGE_RANK ? shape[index] : -1L;
		} catch (OrtException e) {
			return -1L;
		}
	}
}
