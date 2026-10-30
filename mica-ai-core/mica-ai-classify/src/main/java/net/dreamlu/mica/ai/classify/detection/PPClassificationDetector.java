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
package net.dreamlu.mica.ai.classify.detection;

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
import net.dreamlu.mica.ai.classify.config.ClassifyConfig;
import net.dreamlu.mica.ai.classify.model.ClassifyResult;
import net.dreamlu.mica.ai.classify.model.ClassifyTopK;
import net.dreamlu.mica.ai.classify.postprocess.ClassifyPostProcessor;
import net.dreamlu.mica.ai.classify.util.ClassifyImageUtils;
import net.dreamlu.mica.ai.classify.util.ClassifyLabelUtils;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * PULC 分类器（PaddleClas 导出经 paddle2onnx 转换的 ONNX 形态，I/O 实测于 2026-10-30）。
 *
 * <p>ONNX 输入（实测）：
 * <ul>
 *   <li>{@code x}：{@code [N, 3, 224, 224]} float32（动态 batch），
 *       RGB + 短边 256 缩放 + 中心裁剪 224 + ImageNet 归一化</li>
 * </ul>
 * ONNX 输出（实测）：
 * <ul>
 *   <li>{@code softmax_1.tmp_0}：{@code [N, classes]} float32，<b>已内置 softmax</b>
 *       （{@code text_image_orientation} / {@code clarity_assessment} / {@code person_exists}）</li>
 *   <li>{@code sigmoid_2.tmp_0}：{@code [N, 6]} float32，<b>已内置 sigmoid</b>
 *       （{@code table_attribute}，多标签）</li>
 * </ul>
 *
 * <p>输出节点名是 paddle2onnx 从 Paddle 算子名自动派生的，不可硬编码；
 * 定位策略为「先按名称提示（softmax / sigmoid / logits / output / prob）精确匹配，
 * 再子串兜底，最后回落到『唯一的 float [N, classes] 输出』结构判定」。
 *
 * <p>类别索引语义由配套 label_list 文件决定，<b>不凭输出维度猜测</b>：
 * 文件行序即类别索引，重复标签保留，条目数与模型类别数不一致时快速失败。
 *
 * <p>线程安全：ONNX session 线程安全，每次推理新建 1 个 {@code OnnxTensor}；
 * 通常由 {@link net.dreamlu.mica.ai.classify.core.ClassifyEngine} 单例持有。
 */
@Slf4j
public class PPClassificationDetector implements AutoCloseable {

	private static final int IMAGE_RANK = 4;
	private static final int MAX_CLASSES = 1000;

	private final OrtEnvironment environment;
	private final OnnxModelSession session;
	@Getter
	private final ClassifyConfig config;
	@Getter
	private final String[] labels;
	private final String imageInputName;
	private final String scoreOutputName;

	/**
	 * 构造 PULC 分类器，加载 ONNX 模型与 label_list，并解析输入输出张量名。
	 *
	 * @param environment    ONNX 运行环境
	 * @param config         分类配置
	 * @param sessionOptions ONNX 会话选项
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载或张量名解析失败；
	 *                         {@link ErrorCode#NOT_FOUND} label_list 缺失；
	 *                         {@link ErrorCode#ILLEGAL_ARGUMENT} label_list 非法或与模型类别数不一致
	 */
	public PPClassificationDetector(OrtEnvironment environment, ClassifyConfig config,
									OrtSession.SessionOptions sessionOptions) {
		this.environment = environment;
		this.config = config;
		this.labels = ClassifyLabelUtils.readLabels(config.getLabelsPath());
		this.session = new OnnxModelSession(
			environment, config.resolveModelPath(), sessionOptions, "classify");
		this.imageInputName = resolveInputName();
		this.scoreOutputName = resolveOutputName();
		validateImageInputShape();
		validateClassCount();
		log.info("mica-ai-classify 初始化完成: version={} input={} output={} labels={} multiLabel={}",
			config.getModelVersion(), imageInputName, scoreOutputName,
			labels.length, config.isMultiLabel());
	}

	/**
	 * 对图像做分类。
	 *
	 * @param bgr 原图（BGR）；null 或空图时返回 null
	 * @return 分类结果；输入为空时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 预处理、推理或类别数校验失败
	 */
	public ClassifyResult classify(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return null;
		}
		float[] tensorData = ClassifyImageUtils.toChwFloat(
			bgr, config.getInputWidth(), config.getInputHeight(), config.getResizeShort(),
			config.getMean(), config.getStd(), config.getChannelOrder(), config.getInterpolation());

		OnnxTensor inputTensor = null;
		try {
			inputTensor = OnnxTensor.createTensor(environment,
				FloatBuffer.wrap(tensorData),
				new long[]{1, 3, config.getInputHeight(), config.getInputWidth()});
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(imageInputName, inputTensor);
			try (OrtSession.Result result = session.getSession().run(inputs)) {
				return decodeScores(result);
			} catch (OrtException e) {
				throw new MicaAiException(
					ErrorCode.INFERENCE_FAILED, "classify 模型推理失败", e);
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "创建 classify 输入张量失败", e);
		} finally {
			closeQuietly(inputTensor);
		}
	}

	/**
	 * 解码字节数组为 BGR Mat 并跑 {@link #classify(Mat)}，自动 release 临时 Mat。
	 *
	 * @param imageBytes 图像字节（非空）
	 * @return 分类结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public ClassifyResult classifyBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = ClassifyImageUtils.byteArrayToMat(imageBytes);
			return classify(bgr);
		} finally {
			ClassifyImageUtils.releaseAll(bgr);
		}
	}

	private ClassifyResult decodeScores(OrtSession.Result result) throws OrtException {
		Object value = result.get(scoreOutputName).get().getValue();
		float[] raw = asOneDim(value);
		if (raw == null || raw.length == 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"classify 输出 " + scoreOutputName + " 不是 1 维 float 张量");
		}
		if (raw.length != labels.length) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"模型输出类别数 " + raw.length + " 与 label_list 条目数 " + labels.length
					+ " 不一致；请确认模型与 " + config.getLabelsPath() + " 是否成对");
		}
		float[] scores;
		if (config.isOutputIsProbability()) {
			scores = raw;
		} else if (config.isMultiLabel()) {
			scores = ClassifyPostProcessor.sigmoid(raw);
		} else {
			scores = ClassifyPostProcessor.softmax(raw);
		}
		List<ClassifyTopK> topK =
			ClassifyPostProcessor.topK(scores, labels, config.getTopK());
		return new ClassifyResult(topK, scores, labels, config.isMultiLabel());
	}

	private String resolveInputName() {
		Map<String, String> byName = new LinkedHashMap<>();
		try {
			for (String name : session.getSession().getInputNames()) {
				byName.put(name.toLowerCase(Locale.ROOT), name);
			}
		} catch (RuntimeException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 classify 模型输入名失败", e);
		}
		String name = pickByName(byName, "x", "input", "images", "image");
		if (name == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"classify 模型缺少输入节点（期望 x / input / images）");
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"classify 模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		return name;
	}

	private String resolveOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = session.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 classify 模型输出元信息失败", e);
		}
		Map<String, String> byName = new LinkedHashMap<>();
		for (String name : outputInfo.keySet()) {
			byName.put(name.toLowerCase(Locale.ROOT), name);
		}
		String hinted = pickByName(byName, "softmax", "sigmoid", "prob", "logits", "output");
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
				"未能在 classify 模型中定位分类输出（期望 float [N, classes]）");
		}
		log.info("classify 输出节点按结构兜底解析: {}", matched);
		return matched;
	}

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
					"classify 模型 input 应为 4 维，实际 rank=" + shape.length);
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
				ErrorCode.MODEL_LOAD_FAILED, "读取 classify 模型输入形状失败", e);
		}
	}

	/**
	 * 构造期校验 label_list 与模型输出类别数一致（形状动态时留到推理期校验）。
	 */
	private void validateClassCount() {
		try {
			TensorInfo tensor = asTensorInfo(
				session.getSession().getOutputInfo().get(scoreOutputName));
			if (tensor == null) {
				return;
			}
			long[] shape = tensor.getShape();
			if (shape.length == 2 && shape[1] > 0 && shape[1] != labels.length) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"label_list 条目数 " + labels.length + " 与模型类别数 " + shape[1]
						+ " 不一致: " + config.getLabelsPath());
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 classify 模型输出形状失败", e);
		}
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
	 * 类别数（以 label_list 条目数为准）。
	 *
	 * @return 类别数
	 */
	public int classCount() {
		return labels.length;
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
