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
package net.dreamlu.mica.ai.humanseg.detection;

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
import net.dreamlu.mica.ai.humanseg.config.HumanSegConfig;
import net.dreamlu.mica.ai.humanseg.util.HumanSegImageUtils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * PP-HumanSeg 人像分割检测器（opencv_zoo 导出形态，I/O 实测于 2026-10-30）。
 *
 * <p>ONNX 输入：
 * <ul>
 *   <li>{@code x}：{@code [1, 3, 192, 192]} float32，BGR→RGB + CHW +
 *       {@code (x/255-0.5)/0.5}；直接 resize，<b>不做 letterbox</b></li>
 * </ul>
 * ONNX 输出（<b>1 个</b>）：
 * <ul>
 *   <li>{@code save_infer_model/scale_0.tmp_1}：{@code [1, 2, 192, 192]} float32，
 *       已内置 Softmax，两通道和恒为 1</li>
 *   <li>通道 0 = 背景，通道 1 = 人像前景（实测：人像图通道 1 显著大于通道 0）</li>
 * </ul>
 *
 * <p>⚠️ 输出节点名是 Paddle 导出的中间变量名，换一次导出就会变，因此本模块只做
 * 「有且仅有 1 个输出」的校验，不按名字查找；输入节点名按名称提示解析。
 *
 * <p>线程安全：ONNX session 线程安全，每次推理新建 1 个 {@code OnnxTensor}；
 * 通常由 {@link net.dreamlu.mica.ai.humanseg.core.HumanSegEngine} 单例持有。
 */
@Slf4j
public class HumanSegDetector implements AutoCloseable {

	private static final int IMAGE_RANK = 4;
	private static final int BACKGROUND_CHANNEL = 0;
	private static final int FOREGROUND_CHANNEL = 1;

	private final OrtEnvironment environment;
	private final OnnxModelSession session;
	@Getter
	private final HumanSegConfig config;
	private final String imageInputName;
	private final String alphaOutputName;

	/**
	 * 构造人像分割检测器，加载 ONNX 模型并校验输入输出契约。
	 *
	 * @param environment    ONNX 运行环境
	 * @param config         分割配置
	 * @param sessionOptions ONNX 会话选项
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载或契约校验失败
	 */
	public HumanSegDetector(OrtEnvironment environment, HumanSegConfig config,
							OrtSession.SessionOptions sessionOptions) {
		this.environment = environment;
		this.config = config;
		this.session = new OnnxModelSession(environment, config.resolveModelPath(),
			sessionOptions, "humanseg");
		this.imageInputName = resolveInputName();
		this.alphaOutputName = resolveAlphaOutputName();
		validateInputSize();
		log.info("mica-ai-humanseg 初始化完成: version={} input={} output={} size={}x{}",
			config.getModelVersion(), imageInputName, alphaOutputName,
			config.getInputWidth(), config.getInputHeight());
	}

	/**
	 * 计算原尺寸人像 alpha 掩码。
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
		int inputW = config.getInputWidth();
		int inputH = config.getInputHeight();
		float[] tensorData = HumanSegImageUtils.resizeToRgbChwFloat(
			bgr, inputW, inputH, config.getMean(), config.getStd());

		OnnxTensor inputTensor = null;
		try {
			inputTensor = OnnxTensor.createTensor(environment,
				FloatBuffer.wrap(tensorData),
				new long[]{1, 3, inputH, inputW});
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(imageInputName, inputTensor);
			try (OrtSession.Result result = session.getSession().run(inputs)) {
				return extractAlpha(result, origW, origH);
			} catch (OrtException e) {
				throw new MicaAiException(
					ErrorCode.INFERENCE_FAILED, "humanseg 模型推理失败", e);
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "创建 humanseg 输入张量失败", e);
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
			bgr = HumanSegImageUtils.byteArrayToMat(imageBytes);
			return alpha(bgr);
		} finally {
			HumanSegImageUtils.releaseAll(bgr);
		}
	}

	/**
	 * 从模型输出中取前景通道概率，并缩放到原图尺寸。
	 */
	private Mat extractAlpha(OrtSession.Result result, int origW, int origH)
		throws OrtException {
		Object value = result.get(alphaOutputName).get().getValue();
		float[][][][] tensor = asFourDim(value);
		if (tensor == null || tensor.length != 1) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"humanseg 输出 " + alphaOutputName + " 不是 4 维 float 张量");
		}
		float[][][] batch = tensor[0];
		int channels = batch.length;
		int channel;
		if (channels == 1) {
			channel = BACKGROUND_CHANNEL;
		} else if (channels == 2) {
			channel = FOREGROUND_CHANNEL;
		} else {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"humanseg 输出通道数应为 1 或 2，实际 " + channels);
		}
		float[][] plane = batch[channel];
		if (plane.length == 0 || plane[0].length == 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"humanseg 输出 " + alphaOutputName + " 空间维度为空");
		}
		int h = plane.length;
		int w = plane[0].length;
		float[] flat = new float[h * w];
		for (int y = 0; y < h; y++) {
			float[] row = plane[y];
			if (row.length != w) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
					"humanseg 输出 " + alphaOutputName + " 行宽不一致");
			}
			System.arraycopy(row, 0, flat, y * w, w);
		}

		Mat raw = new Mat(h, w, CvType.CV_32FC1);
		Mat resized = null;
		try {
			raw.put(0, 0, flat);
			// 尺寸已一致时跳过缩放，避免无谓的重采样损失
			if (w == origW && h == origH) {
				Mat copy = new Mat();
				raw.copyTo(copy);
				return copy;
			}
			resized = HumanSegImageUtils.resizeMask(raw, origW, origH);
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
				ErrorCode.MODEL_LOAD_FAILED, "读取 humanseg 模型输入名失败", e);
		}
		String name = pickByName(byName, "x", "input.1", "input", "images", "image");
		if (name == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"humanseg 模型缺少输入节点（期望 x / input.1 / input / images）");
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"humanseg 模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		return name;
	}

	/**
	 * 定位前景输出节点：PP-HumanSeg 只有 1 个输出，直接取首个并校验形态。
	 */
	private String resolveAlphaOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = session.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 humanseg 模型输出元信息失败", e);
		}
		if (outputInfo.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"humanseg 模型应有且仅有 1 个输出，实际 " + outputInfo.size() + " 个");
		}
		String name = outputInfo.keySet().iterator().next();
		TensorInfo tensor = asTensorInfo(outputInfo.get(name));
		if (tensor == null || tensor.type != OnnxJavaType.FLOAT) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"humanseg 输出应为 float32 张量，实际 " + outputInfo.get(name));
		}
		long[] shape = tensor.getShape();
		if (shape.length != IMAGE_RANK || shape[BACKGROUND_CHANNEL] != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"humanseg 输出应为 [1,C,H,W]，实际 " + java.util.Arrays.toString(shape));
		}
		return name;
	}

	/**
	 * 校验模型输入宽高与配置一致，否则预处理后的张量与模型形状对不上。
	 */
	private void validateInputSize() {
		try {
			ValueInfo info = session.getSession().getInputInfo().get(imageInputName).getInfo();
			TensorInfo tensor = info instanceof TensorInfo ? (TensorInfo) info : null;
			if (tensor == null) {
				return;
			}
			long[] shape = tensor.getShape();
			if (shape.length != IMAGE_RANK) {
				throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
					"humanseg 模型 input 应为 4 维，实际 rank=" + shape.length);
			}
			long modelH = shape[IMAGE_RANK - 2];
			long modelW = shape[IMAGE_RANK - 1];
			if (modelH > 0 && modelW > 0
				&& (modelH != config.getInputHeight() || modelW != config.getInputWidth())) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"inputWidth/inputHeight 必须等于模型输入宽高 " + modelW + "x" + modelH
						+ "，当前为 " + config.getInputWidth() + "x" + config.getInputHeight());
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 humanseg 模型输入形状失败", e);
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
	 * 便于测试断言：模型输入宽度（NCHW 最后一维）。
	 *
	 * @return 输入宽度；读取失败返回 -1
	 */
	public int modelInputWidth() {
		return modelInputSize(IMAGE_RANK - 1);
	}

	/**
	 * 便于测试断言：模型输入高度（NCHW 倒数第二维）。
	 *
	 * @return 输入高度；读取失败返回 -1
	 */
	public int modelInputHeight() {
		return modelInputSize(IMAGE_RANK - 2);
	}

	private int modelInputSize(int dim) {
		try {
			TensorInfo tensor = asTensorInfo(
				session.getSession().getInputInfo().get(imageInputName));
			if (tensor == null) {
				return -1;
			}
			long[] shape = tensor.getShape();
			return shape.length == IMAGE_RANK && shape[dim] > 0 ? (int) shape[dim] : -1;
		} catch (OrtException e) {
			return -1;
		}
	}

	/**
	 * 便于测试断言：模型输出节点数（PP-HumanSeg 应为 1）。
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
