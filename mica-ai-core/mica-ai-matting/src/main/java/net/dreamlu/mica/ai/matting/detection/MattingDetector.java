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
package net.dreamlu.mica.ai.matting.detection;

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
import net.dreamlu.mica.ai.matting.config.MattingConfig;
import net.dreamlu.mica.ai.matting.config.MattingOutputSelect;
import net.dreamlu.mica.ai.matting.util.MattingImageUtils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * U²-Net 族抠图检测器（rembg 导出形态，I/O 实测于 2026-09-21）。
 *
 * <p>ONNX 输入：
 * <ul>
 *   <li>{@code input.1}：{@code [1, 3, 320, 320]} float32，BGR→RGB + CHW + {@code (x/255-mean)/std}</li>
 * </ul>
 * ONNX 输出（<b>7 个</b>，形状全部为 {@code [1, 1, 320, 320]} float32）：
 * <ul>
 *   <li>索引 0 即 d0，是被 Sigmoid 压到 {@code [0,1]} 的融合显著性掩码，<b>本模块唯一消费的节点</b></li>
 *   <li>索引 1..6（d1..d6）是训练期的中间监督（deep supervision）输出，
 *       与 d0 形状、值域完全一致，推理阶段必须忽略</li>
 * </ul>
 *
 * <p>⚠️ <b>输出节点名不可依赖</b>：rembg 的导出是 torch.onnx.export 的产物，输出名是数字
 * （实测 {@code 1959..1965}），换一次导出就会变。因此定位 d0 的策略被显式化为
 * {@link net.dreamlu.mica.ai.matting.config.MattingOutputSelect}，可配置（默认 AUTO）：
 * 先按名称提示匹配，再回落到「校验 7 个同形输出 + 取首个」。
 *
 * <p>输入节点名同样按名称提示解析（{@code input.1} / {@code input} / {@code images}，
 * 含子串兜底），保证换导出 / 换文件名不会静默错位。
 *
 * <p><b>模型可插拔</b>：本检测器不绑定具体模型，凡「1 输入 + 7 输出 + d0 语义」的
 * rembg U²-Net 族模型（u2netp / u2net / u2net_human_seg 等）均可直接加载。
 *
 * <p>线程安全：ONNX session 线程安全，每次推理新建 1 个 {@code OnnxTensor}；
 * 通常由 {@link net.dreamlu.mica.ai.matting.MattingEngine} 单例持有。
 */
@Slf4j
public class MattingDetector implements AutoCloseable {

	private static final int IMAGE_RANK = 4;
	private static final int CHANNEL_DIM = 1;

	private final OrtEnvironment environment;
	private final OnnxModelSession session;
	@Getter
	private final MattingConfig config;
	private final String imageInputName;
	private final String alphaOutputName;

	/**
	 * 构造抠图检测器，加载 ONNX 模型并解析输入输出张量名。
	 *
	 * @param environment    ONNX 运行环境
	 * @param config         抠图配置
	 * @param sessionOptions ONNX 会话选项
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载或张量名解析失败
	 */
	public MattingDetector(OrtEnvironment environment, MattingConfig config,
						   OrtSession.SessionOptions sessionOptions) {
		this.environment = environment;
		this.config = config;
		this.session = new OnnxModelSession(
			environment, config.resolveModelPath(),
			sessionOptions, "matting");
		this.imageInputName = resolveInputName();
		this.alphaOutputName = resolveAlphaOutputName();
		validateImageInputSize();
		log.info("mica-ai-matting 初始化完成: version={} input={} alpha={} select={}",
			config.getModelVersion(), imageInputName, alphaOutputName, config.getOutputSelect());
	}

	/**
	 * 计算原尺寸 alpha 掩码。
	 *
	 * @param bgr 原图 BGR Mat；null 或空图时返回 null
	 * @return 与原图同尺寸的单通道 {@code CV_32FC1} 掩码（取值 [0,1]），调用方负责 release；输入为空时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 预处理或 ONNX 推理失败
	 */
	public Mat alpha(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return null;
		}
		int origW = bgr.cols();
		int origH = bgr.rows();
		int inputSize = config.getInputSize();
		float[] tensorData = MattingImageUtils.resizeToRgbChwFloat(
			bgr, inputSize, config.getMean(), config.getStd());

		OnnxTensor inputTensor = null;
		try {
			inputTensor = OnnxTensor.createTensor(environment,
				FloatBuffer.wrap(tensorData),
				new long[]{1, 3, inputSize, inputSize});
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(imageInputName, inputTensor);
			try (OrtSession.Result result = session.getSession().run(inputs)) {
				return extractAlpha(result, origW, origH);
			} catch (OrtException e) {
				throw new MicaAiException(
					ErrorCode.INFERENCE_FAILED, "matting 模型推理失败", e);
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "创建 matting 输入张量失败", e);
		} finally {
			closeQuietly(inputTensor);
		}
	}

	/**
	 * 解码字节数组为 BGR Mat 并跑 {@link #alpha(Mat)}，自动 release 临时 Mat。
	 *
	 * @param imageBytes 图像字节（非空）
	 * @return 原尺寸单通道 {@code CV_32FC1} 掩码，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public Mat alphaBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = MattingImageUtils.byteArrayToMat(imageBytes);
			return alpha(bgr);
		} finally {
			MattingImageUtils.releaseAll(bgr);
		}
	}

	private Mat extractAlpha(OrtSession.Result result, int origW, int origH) throws OrtException {
		Object value = result.get(alphaOutputName).get().getValue();
		float[][] alpha = asTwoDim(value);
		if (alpha == null || alpha.length == 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"matting 输出 " + alphaOutputName + " 不是 2 维 float 张量");
		}
		int h = alpha.length;
		int w = alpha[0].length;
		float[] flat = new float[h * w];
		for (int y = 0; y < h; y++) {
			float[] row = alpha[y];
			if (row.length != w) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
					"matting 输出 " + alphaOutputName + " 行宽不一致");
			}
			System.arraycopy(row, 0, flat, y * w, w);
		}

		Mat raw = new Mat(h, w, CvType.CV_32FC1);
		Mat resized = null;
		try {
			raw.put(0, 0, flat);
			if (config.isMinMaxNormalize()) {
				MattingImageUtils.minMaxNormalize(raw);
			}
			// 尺寸已一致时跳过缩放，避免无谓的重采样损失
			if (w == origW && h == origH) {
				Mat copy = new Mat();
				raw.copyTo(copy);
				return copy;
			}
			resized = MattingImageUtils.resizeMask(raw, origW, origH, config.getInterpolation());
			Mat copy = new Mat();
			resized.copyTo(copy);
			return copy;
		} finally {
			raw.release();
			if (resized != null) {
				resized.release();
			}
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
				ErrorCode.MODEL_LOAD_FAILED, "读取 matting 模型输入名失败", e);
		}
		String name = pickByName(byName, "input.1", "input", "images", "image");
		if (name == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"matting 模型缺少输入节点（期望 input.1 / input / images）");
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"matting 模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		return name;
	}

	/**
	 * 定位显著性输出节点 d0，行为由 {@link MattingOutputSelect} 配置驱动。
	 *
	 * <p>{@code AUTO}：先按名称提示（d0 / alpha / saliency / mask）解析；
	 * 无提示时校验「输出数 == 期望值」且取首个 float {@code [1,1,H,W]} 输出。
	 */
	private String resolveAlphaOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = session.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 matting 模型输出元信息失败", e);
		}
		MattingOutputSelect select = config.getOutputSelect();
		String[] names = outputInfo.keySet().toArray(new String[0]);
		String hinted = select.resolveByName(names);
		if (hinted != null) {
			return hinted;
		}
		// 无名称提示（或策略要求按序取）：依赖 U²-Net 族特征——7 个同形的单通道输出
		select.validate(outputInfo.size());
		String first = null;
		for (Map.Entry<String, NodeInfo> entry : outputInfo.entrySet()) {
			TensorInfo tensor = asTensorInfo(entry.getValue());
			if (tensor == null || tensor.type != OnnxJavaType.FLOAT) {
				continue;
			}
			long[] shape = tensor.getShape();
			if (shape.length != IMAGE_RANK || shape[CHANNEL_DIM] != 1) {
				continue;
			}
			if (first == null) {
				first = entry.getKey();
			}
		}
		if (first == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"未能在 matting 模型中定位显著性输出（期望 float [1,1,H,W]）");
		}
		log.info("matting 输出节点按策略 {} 取首个输出: {}", select, first);
		return first;
	}

	/**
	 * 校验模型输入边长与 {@code inputSize} 一致，否则缩放结果与模型张量形状对不上。
	 */
	private void validateImageInputSize() {
		try {
			NodeInfo nodeInfo = session.getSession().getInputInfo().get(imageInputName);
			TensorInfo tensor = asTensorInfo(nodeInfo);
			if (tensor == null) {
				return;
			}
			long[] shape = tensor.getShape();
			if (shape.length != IMAGE_RANK) {
				throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
					"matting 模型 input 应为 4 维，实际 rank=" + shape.length);
			}
			long modelH = shape[IMAGE_RANK - 2];
			long modelW = shape[IMAGE_RANK - 1];
			if (modelH > 0 && modelW > 0
				&& (modelH != config.getInputSize() || modelW != config.getInputSize())) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"inputSize 必须等于模型输入边长 " + modelH + "，当前为 "
						+ config.getInputSize());
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 matting 模型输入形状失败", e);
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

	private static float[][] asTwoDim(Object value) {
		if (value instanceof float[][]) {
			return (float[][]) value;
		}
		if (value instanceof float[][][]) {
			float[][][] three = (float[][][]) value;
			return three.length > 0 ? three[0] : null;
		}
		if (value instanceof float[][][][]) {
			float[][][][] four = (float[][][][]) value;
			return four.length > 0 && four[0].length > 0 ? four[0][0] : null;
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
	 * 便于测试断言：模型输出节点总数（u2netp 应为 7）。
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

	/**
	 * 便于测试断言：模型输入边长（NCHW 的最后两维）。
	 *
	 * @return 输入边长；读取失败返回 -1
	 */
	public int modelInputSize() {
		try {
			TensorInfo tensor = asTensorInfo(
				session.getSession().getInputInfo().get(imageInputName));
			if (tensor == null) {
				return -1;
			}
			long[] shape = tensor.getShape();
			return shape.length == IMAGE_RANK && shape[IMAGE_RANK - 1] > 0
				? (int) shape[IMAGE_RANK - 1] : -1;
		} catch (OrtException e) {
			return -1;
		}
	}
}
