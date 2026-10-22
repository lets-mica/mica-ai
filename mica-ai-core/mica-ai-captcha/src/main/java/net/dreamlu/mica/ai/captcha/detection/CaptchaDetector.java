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
package net.dreamlu.mica.ai.captcha.detection;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.captcha.charset.CharsetManager;
import net.dreamlu.mica.ai.captcha.config.CaptchaConfig;
import net.dreamlu.mica.ai.captcha.model.CaptchaResult;
import net.dreamlu.mica.ai.captcha.util.CaptchaImageUtils;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;

import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * 通用验证码识别器（CRNN + CTC 解码）。
 *
 * <p>ONNX 输入（{@code ddddocr} 官方导出，实测 2026-10-22）：
 * <ul>
 *   <li>{@code input1}：{@code [1, 1, 64, image_width]} float32（高度固定 64，宽度动态）</li>
 * </ul>
 * ONNX 输出：
 * <ul>
 *   <li>{@code 387}：运行时形状 {@code [seqlen, 1, num_classes]} float32，<b>原始 logits</b>
 *       （未 softmax）</li>
 * </ul>
 *
 * <p><b>⚠️ 输出节点的声明形状不可信</b>：该模型是动态量化版本（{@code ConvInteger} +
 * {@code DynamicQuantizeLinear}），ONNX 元信息里输出形状写的是 {@code [1, seqlen]}，
 * 但运行时实际返回 {@code [seqlen, 1, num_classes]}（实测 {@code [23, 1, 8210]}）。
 * 因此本类<b>不读声明形状</b>，一律按运行时返回的数组维度归一化，
 * 并同时兼容 {@code [seq][1][cls]} / {@code [1][seq][cls]} / {@code [seq][cls]} 三种布局。
 *
 * <p>线程安全：ONNX session 线程安全，每次推理新建 1 个 {@code OnnxTensor}；
 * 通常由 {@link net.dreamlu.mica.ai.captcha.CaptchaEngine} 单例持有。
 */
@Slf4j
public class CaptchaDetector implements AutoCloseable {

	private static final String[] INPUT_NAME_HINTS = {"input1", "input", "x", "image", "images"};

	private final OrtEnvironment environment;
	private final OnnxModelSession session;
	@Getter
	private final CaptchaConfig config;
	private final CharsetManager charset;
	private final String inputName;
	private final String outputName;

	/**
	 * 构造识别器，加载 ONNX 模型与外部字符集。
	 *
	 * @param environment    ONNX 运行环境
	 * @param config         验证码识别配置
	 * @param sessionOptions ONNX 会话选项
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载或字符集加载失败
	 */
	public CaptchaDetector(OrtEnvironment environment, CaptchaConfig config,
						   OrtSession.SessionOptions sessionOptions) {
		this.environment = environment;
		this.config = config;
		this.session = new OnnxModelSession(
			environment, config.resolveModelPath(), sessionOptions, "captcha");
		this.charset = CharsetManager.load(config.getCharsetPath());
		this.inputName = resolveInputName();
		this.outputName = resolveOutputName();
		validateInputHeight();
		log.info("mica-ai-captcha 初始化完成: model={} input={} output={} 类别数={}",
			config.getModelPath(), inputName, outputName, charset.size());
	}

	/**
	 * 识别验证码图片。
	 *
	 * @param image 待识别图片
	 * @return 识别结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 预处理或推理失败；
	 *                         字符集与模型类别数不匹配时同样抛出
	 */
	public CaptchaResult recognize(BufferedImage image) {
		if (image == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "图片不能为空");
		}
		CaptchaImageUtils.ImageTensor tensor = CaptchaImageUtils.prepareForOnnx(
			image, config.getInputHeight(), config.getInterpolation(),
			config.getNormalization(), config.isPngFix());

		OnnxTensor inputTensor = null;
		try {
			inputTensor = OnnxTensor.createTensor(environment,
				FloatBuffer.wrap(tensor.getData()),
				new long[]{1, 1, tensor.getHeight(), tensor.getWidth()});
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(inputName, inputTensor);
			try (OrtSession.Result result = session.getSession().run(inputs)) {
				return decode(extractLogits(result));
			} catch (OrtException e) {
				throw new MicaAiException(
					ErrorCode.INFERENCE_FAILED, "captcha 模型推理失败", e);
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "创建 captcha 输入张量失败", e);
		} finally {
			closeQuietly(inputTensor);
		}
	}

	/**
	 * 取模型输出并归一化为 {@code [seqlen][num_classes]} 的二维 logits。
	 */
	private float[][] extractLogits(OrtSession.Result result) throws OrtException {
		Object value = result.get(outputName).get().getValue();
		float[][] matrix = toMatrix(value);
		if (matrix.length == 0 || matrix[0].length == 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"captcha 模型输出为空");
		}
		return matrix;
	}

	/**
	 * CTC 贪心解码：逐时刻取 argmax，折叠连续重复，跳过 blank。
	 */
	private CaptchaResult decode(float[][] logits) {
		int numClasses = logits[0].length;
		if (numClasses != charset.size()) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"字符集大小 (" + charset.size() + ") 与模型输出类别数 (" + numClasses
					+ ") 不匹配；字符集文件必须与模型成对使用");
		}
		StringBuilder text = new StringBuilder();
		List<Float> confidences = new ArrayList<>();
		int previous = -1;
		for (float[] row : logits) {
			int best = argmax(row);
			if (best != previous && best != CharsetManager.BLANK_INDEX) {
				text.append(charset.charAt(best));
				confidences.add(softmaxAt(row, best));
			}
			previous = best;
		}
		float[] charConfidences = new float[confidences.size()];
		float sum = 0f;
		for (int i = 0; i < confidences.size(); i++) {
			charConfidences[i] = confidences.get(i);
			sum += charConfidences[i];
		}
		float confidence = confidences.isEmpty() ? 0f : sum / confidences.size();
		return new CaptchaResult(text.toString(), confidence, charConfidences);
	}

	/**
	 * 把 ONNX 运行时输出归一化为 {@code [seqlen][num_classes]}。
	 *
	 * <p>兼容三种布局：
	 * <ul>
	 *   <li>{@code [seq][1][cls]}（本模型实测形态）</li>
	 *   <li>{@code [1][seq][cls]}（ddddocr 老模型形态）</li>
	 *   <li>{@code [seq][cls]}</li>
	 * </ul>
	 */
	private static float[][] toMatrix(Object value) {
		if (value instanceof float[][][]) {
			float[][][] tensor = (float[][][]) value;
			int a = tensor.length;
			if (a == 0 || tensor[0].length == 0 || tensor[0][0].length == 0) {
				return new float[0][0];
			}
			int b = tensor[0].length;
			int c = tensor[0][0].length;
			if (c == 1) {
				float[][] out = new float[a][b];
				for (int i = 0; i < a; i++) {
					for (int j = 0; j < b; j++) {
						out[i][j] = tensor[i][j][0];
					}
				}
				return out;
			}
			if (b == 1) {
				float[][] out = new float[a][c];
				for (int i = 0; i < a; i++) {
					System.arraycopy(tensor[i][0], 0, out[i], 0, c);
				}
				return out;
			}
			if (a == 1) {
				return tensor[0];
			}
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"无法解析 captcha 模型输出形状 [" + a + "][" + b + "][" + c + "]");
		}
		if (value instanceof float[][]) {
			return (float[][]) value;
		}
		if (value instanceof float[]) {
			return new float[][]{(float[]) value};
		}
		throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
			"不支持的 captcha 模型输出类型: "
				+ (value == null ? "null" : value.getClass().getName()));
	}

	private static int argmax(float[] row) {
		int best = 0;
		float bestValue = row[0];
		for (int i = 1; i < row.length; i++) {
			if (row[i] > bestValue) {
				bestValue = row[i];
				best = i;
			}
		}
		return best;
	}

	/**
	 * 计算指定类别的 softmax 概率（数值稳定版）。
	 */
	private static float softmaxAt(float[] row, int index) {
		float max = row[0];
		for (float v : row) {
			if (v > max) {
				max = v;
			}
		}
		double sum = 0d;
		double[] exp = new double[row.length];
		for (int i = 0; i < row.length; i++) {
			exp[i] = Math.exp(row[i] - max);
			sum += exp[i];
		}
		return (float) (exp[index] / sum);
	}

	/**
	 * 解析模型输入节点名：先按名称提示匹配，再回落到「唯一的 float 输入」。
	 */
	private String resolveInputName() {
		Map<String, String> byName = new LinkedHashMap<>();
		try {
			for (String name : session.getSession().getInputNames()) {
				byName.put(name.toLowerCase(Locale.ROOT), name);
			}
		} catch (RuntimeException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 captcha 模型输入名失败", e);
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"captcha 模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		String name = pickByName(byName, INPUT_NAME_HINTS);
		return name != null ? name : byName.values().iterator().next();
	}

	/**
	 * 解析模型输出节点名。
	 *
	 * <p>本模型的输出节点名为 {@code 387}（自动编号），无法按语义命名匹配，
	 * 因此以「唯一输出」为准；多输出时按名称提示回落。
	 */
	private String resolveOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = session.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 captcha 模型输出元信息失败", e);
		}
		if (outputInfo.size() == 1) {
			return outputInfo.keySet().iterator().next();
		}
		Map<String, String> byName = new LinkedHashMap<>();
		for (Map.Entry<String, NodeInfo> entry : outputInfo.entrySet()) {
			byName.put(entry.getKey().toLowerCase(Locale.ROOT), entry.getKey());
		}
		String name = pickByName(byName, "output", "logits", "prob", "softmax", "387");
		if (name == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"未能定位 captcha 模型输出节点，候选: " + outputInfo.keySet());
		}
		return name;
	}

	/**
	 * 校验模型输入高度与配置一致（宽度是动态维度，不校验）。
	 */
	public void validateInputHeight() {
		try {
			NodeInfo nodeInfo = session.getSession().getInputInfo().get(inputName);
			TensorInfo tensor = asTensorInfo(nodeInfo);
			if (tensor == null) {
				return;
			}
			long[] shape = tensor.getShape();
			if (shape.length != 4) {
				throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
					"captcha 模型 input 应为 4 维，实际 rank=" + shape.length);
			}
			long modelHeight = shape[2];
			if (modelHeight > 0 && modelHeight != config.getInputHeight()) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"inputHeight 必须等于模型输入高度 " + modelHeight + "，当前为 "
						+ config.getInputHeight());
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 captcha 模型输入形状失败", e);
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
			for (Map.Entry<String, String> entry : meta.entrySet()) {
				if (entry.getKey().contains(hint.toLowerCase(Locale.ROOT))) {
					return entry.getValue();
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
	 * 便于测试断言：模型输入高度（NCHW 倒数第二维）。
	 *
	 * @return 输入高度；读取失败返回 -1
	 */
	public int modelInputHeight() {
		try {
			TensorInfo tensor = asTensorInfo(
				session.getSession().getInputInfo().get(inputName));
			if (tensor == null) {
				return -1;
			}
			long[] shape = tensor.getShape();
			return shape.length == 4 && shape[2] > 0 ? (int) shape[2] : -1;
		} catch (OrtException e) {
			return -1;
		}
	}

	/**
	 * 便于测试断言：字符集类别数（含 blank），应与模型输出类别数一致。
	 *
	 * @return 类别数
	 */
	public int charsetSize() {
		return charset.size();
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
}
