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
package net.dreamlu.mica.ai.enhance.detection;

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
import net.dreamlu.mica.ai.enhance.config.EnhanceConfig;
import net.dreamlu.mica.ai.enhance.util.EnhanceImageUtils;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Real-ESRGAN（SRVGGNetCompact）超分检测器（I/O 实测于 2026-10-30）。
 *
 * <p>ONNX 输入：
 * <ul>
 *   <li>{@code input}：{@code [1, 3, H, W]} float32，<b>H / W 为动态维度</b>；
 *       BGR→RGB + CHW + {@code /255}（<b>无 mean / std</b>）</li>
 * </ul>
 * ONNX 输出（<b>1 个</b>）：
 * <ul>
 *   <li>{@code output}：{@code [1, 3, H*4, W*4]} float32，值域可轻微越界（实测 -0.03 ~ 1.05），
 *       转 8bit 前需 clip 到 {@code [0,1]}</li>
 * </ul>
 *
 * <p>⚠️ 输入是动态尺寸，因此不做「配置宽高 == 模型宽高」校验；改为在每次推理后校验
 * 「输出宽高 == 输入宽高 × {@link EnhanceConfig#getScale()}」，模型与 {@code scale}
 * 不匹配时快速失败。
 *
 * <p>线程安全：ONNX session 线程安全，每次推理新建 1 个 {@code OnnxTensor}；
 * 通常由 {@link net.dreamlu.mica.ai.enhance.core.EnhanceEngine} 单例持有。
 */
@Slf4j
public class EnhanceDetector implements AutoCloseable {

	private static final int IMAGE_RANK = 4;

	private final OrtEnvironment environment;
	private final OnnxModelSession session;
	@Getter
	private final EnhanceConfig config;
	private final String imageInputName;
	private final String imageOutputName;

	/**
	 * 构造超分检测器，加载 ONNX 模型并校验输入输出契约。
	 *
	 * @param environment    ONNX 运行环境
	 * @param config         超分配置
	 * @param sessionOptions ONNX 会话选项
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载或契约校验失败
	 */
	public EnhanceDetector(OrtEnvironment environment, EnhanceConfig config,
						   OrtSession.SessionOptions sessionOptions) {
		this.environment = environment;
		this.config = config;
		this.session = new OnnxModelSession(environment, config.resolveModelPath(),
			sessionOptions, "enhance");
		this.imageInputName = resolveInputName();
		this.imageOutputName = resolveOutputName();
		log.info("mica-ai-enhance 初始化完成: version={} input={} output={} scale={}x tile={} overlap={}",
			config.getModelVersion(), imageInputName, imageOutputName,
			config.getScale(), config.getTileSize(), config.getTileOverlap());
	}

	/**
	 * 对单张 BGR 图做一次推理并放大（不做 tile 切分）。
	 *
	 * @param bgr BGR 三通道 Mat；null 或空图时返回 null
	 * @return {@code scale} 倍尺寸的 BGR Mat，调用方负责 release；输入为空时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败或输出尺寸与 {@code scale} 不符
	 */
	public Mat upscale(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return null;
		}
		int width = bgr.cols();
		int height = bgr.rows();
		float[] chw = EnhanceImageUtils.bgrToRgbChwFloat(bgr);

		OnnxTensor inputTensor = null;
		try {
			inputTensor = OnnxTensor.createTensor(environment,
				FloatBuffer.wrap(chw),
				new long[]{1, 3, height, width});
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(imageInputName, inputTensor);
			try (OrtSession.Result result = session.getSession().run(inputs)) {
				return decode(result, width, height);
			} catch (OrtException e) {
				throw new MicaAiException(
					ErrorCode.INFERENCE_FAILED, "enhance 模型推理失败", e);
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "创建 enhance 输入张量失败", e);
		} finally {
			closeQuietly(inputTensor);
		}
	}

	private Mat decode(OrtSession.Result result, int inputWidth, int inputHeight)
		throws OrtException {
		Object value = result.get(imageOutputName).get().getValue();
		float[][][][] tensor = asFourDim(value);
		if (tensor == null || tensor.length != 1) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"enhance 输出 " + imageOutputName + " 不是 4 维 float 张量");
		}
		float[][][] batch = tensor[0];
		if (batch.length != 3) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"enhance 输出通道数应为 3，实际 " + batch.length);
		}
		int outHeight = batch[0].length;
		int outWidth = outHeight == 0 ? 0 : batch[0][0].length;
		if (outHeight == 0 || outWidth == 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"enhance 输出 " + imageOutputName + " 空间维度为空");
		}
		verifyScale(inputWidth, inputHeight, outWidth, outHeight);

		int plane = outWidth * outHeight;
		float[] flat = new float[plane * 3];
		for (int c = 0; c < 3; c++) {
			float[][] channel = batch[c];
			for (int y = 0; y < outHeight; y++) {
				float[] row = channel[y];
				if (row.length != outWidth) {
					throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
						"enhance 输出 " + imageOutputName + " 行宽不一致");
				}
				System.arraycopy(row, 0, flat, c * plane + y * outWidth, outWidth);
			}
		}
		return EnhanceImageUtils.chwFloatToBgrMat(flat, outWidth, outHeight);
	}

	private void verifyScale(int inputWidth, int inputHeight, int outWidth, int outHeight) {
		int scale = config.getScale();
		if (outWidth != inputWidth * scale || outHeight != inputHeight * scale) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"enhance 模型输出 " + outWidth + "x" + outHeight
					+ " 与输入 " + inputWidth + "x" + inputHeight
					+ " 不满足 scale=" + scale + "，请检查 scale 配置");
		}
	}

	private String resolveInputName() {
		Map<String, String> byName = new LinkedHashMap<>();
		try {
			for (String name : session.getSession().getInputNames()) {
				byName.put(name.toLowerCase(Locale.ROOT), name);
			}
		} catch (RuntimeException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 enhance 模型输入名失败", e);
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"enhance 模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		String name = pickByName(byName, "input", "images", "image", "x", "img");
		if (name == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"enhance 模型缺少输入节点（期望 input / images / image / x）");
		}
		return name;
	}

	private String resolveOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = session.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 enhance 模型输出元信息失败", e);
		}
		if (outputInfo.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"enhance 模型应有且仅有 1 个输出，实际 " + outputInfo.size() + " 个");
		}
		String name = outputInfo.keySet().iterator().next();
		TensorInfo tensor = asTensorInfo(outputInfo.get(name));
		if (tensor == null || tensor.type != OnnxJavaType.FLOAT) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"enhance 输出应为 float32 张量，实际 " + outputInfo.get(name));
		}
		long[] shape = tensor.getShape();
		if (shape.length != IMAGE_RANK || shape[1] != 3) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"enhance 输出应为 [1,3,H,W]，实际 " + java.util.Arrays.toString(shape));
		}
		return name;
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

	private static float[][][][] asFourDim(Object value) {
		if (value instanceof float[][][][]) {
			return (float[][][][]) value;
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
	 * 便于测试断言：模型输出节点数（应为 1）。
	 *
	 * @return 输出节点个数；读取失败返回 -1
	 */
	public int outputCount() {
		try {
			return session.getSession().getOutputInfo().size();
		} catch (OrtException e) {
			return -1;
		}
	}
}
