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
package net.dreamlu.mica.ai.docalign.detection;

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
import net.dreamlu.mica.ai.docalign.config.DocAlignConfig;
import net.dreamlu.mica.ai.docalign.model.DocAlignResult;
import net.dreamlu.mica.ai.docalign.model.DocCorner;
import net.dreamlu.mica.ai.docalign.util.DocAlignImageUtils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * DocAligner 文档角点检测器（I/O 实测于 2026-10-30）。
 *
 * <p>ONNX 输入：{@code img}，{@code [1,3,256,256]} float32，
 * 直接 resize + BGR→RGB + {@code /255}（<b>无 mean/std</b>）。
 *
 * <p>ONNX 输出：{@code heatmap}，{@code [1,4,128,128]} float32，
 * 图内<b>已内置 Sigmoid</b>（值域 0~1），4 个通道依次是
 * <b>左上 / 右上 / 右下 / 左下</b>（已用合成斜拍文档实测确认）。
 *
 * <p>后处理（对齐上游 {@code postprocess}）：把每个通道的热力图 resize 到原图尺寸 →
 * 按 {@code heatmapThreshold} 二值化 → 取面积最大连通域的质心作为角点。
 *
 * <p>线程安全：ONNX session 线程安全，每次推理新建 1 个 {@code OnnxTensor}；
 * 通常由 {@link net.dreamlu.mica.ai.docalign.core.DocAlignEngine} 单例持有。
 */
@Slf4j
public class DocAlignDetector implements AutoCloseable {

	private static final int IMAGE_RANK = 4;
	private static final int EXPECTED_CORNERS = 4;

	private final OrtEnvironment environment;
	private final OnnxModelSession session;
	@Getter
	private final DocAlignConfig config;
	private final String imageInputName;
	private final String heatmapOutputName;

	/**
	 * 构造角点检测器，加载 ONNX 模型并校验输入输出契约。
	 *
	 * @param environment    ONNX 运行环境
	 * @param config         角点检测配置
	 * @param sessionOptions ONNX 会话选项
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载或契约校验失败
	 */
	public DocAlignDetector(OrtEnvironment environment, DocAlignConfig config,
							OrtSession.SessionOptions sessionOptions) {
		this.environment = environment;
		this.config = config;
		this.session = new OnnxModelSession(environment, config.resolveModelPath(),
			sessionOptions, "docalign");
		this.imageInputName = resolveInputName();
		this.heatmapOutputName = resolveHeatmapOutputName();
		validateInputSize();
		log.info("mica-ai-docalign 初始化完成: version={} input={} output={} size={}x{} threshold={}",
			config.getModelVersion(), imageInputName, heatmapOutputName,
			config.getInputWidth(), config.getInputHeight(), config.getHeatmapThreshold());
	}

	/**
	 * 检测文档四角。
	 *
	 * @param bgr 原图 BGR Mat；null 或空图时返回 null
	 * @return 4 个角点（原图坐标系，左上 / 右上 / 右下 / 左下）；输入为空时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 预处理或 ONNX 推理失败
	 */
	public DocAlignResult corners(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return null;
		}
		int origW = bgr.cols();
		int origH = bgr.rows();
		int inputW = config.getInputWidth();
		int inputH = config.getInputHeight();
		float[] tensorData = DocAlignImageUtils.resizeToRgbChwFloat(bgr, inputW, inputH);

		OnnxTensor inputTensor = null;
		try {
			inputTensor = OnnxTensor.createTensor(environment,
				FloatBuffer.wrap(tensorData),
				new long[]{1, 3, inputH, inputW});
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(imageInputName, inputTensor);
			try (OrtSession.Result result = session.getSession().run(inputs)) {
				return decode(result, origW, origH);
			} catch (OrtException e) {
				throw new MicaAiException(
					ErrorCode.INFERENCE_FAILED, "docalign 模型推理失败", e);
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "创建 docalign 输入张量失败", e);
		} finally {
			closeQuietly(inputTensor);
		}
	}

	/**
	 * 解码字节数组为 BGR Mat 并跑 {@link #corners(Mat)}，自动 release 临时 Mat。
	 *
	 * @param imageBytes 图像字节（非空）
	 * @return 4 个角点；解码失败抛异常
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public DocAlignResult cornersBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = DocAlignImageUtils.byteArrayToMat(imageBytes);
			return corners(bgr);
		} finally {
			DocAlignImageUtils.releaseAll(bgr);
		}
	}

	/**
	 * 把 4 个热力图通道逐个还原成原尺寸角点。
	 */
	private DocAlignResult decode(OrtSession.Result result, int origW, int origH)
		throws OrtException {
		Object value = result.get(heatmapOutputName).get().getValue();
		float[][][][] tensor = asFourDim(value);
		if (tensor == null || tensor.length != 1) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"docalign 输出 " + heatmapOutputName + " 不是 4 维 float 张量");
		}
		float[][][] batch = tensor[0];
		if (batch.length != EXPECTED_CORNERS) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"docalign 输出应包含 " + EXPECTED_CORNERS + " 个角点通道，实际 " + batch.length);
		}
		List<DocCorner> corners = new ArrayList<>(EXPECTED_CORNERS);
		for (int c = 0; c < EXPECTED_CORNERS; c++) {
			Mat heatmap = toMat(batch[c]);
			Mat resized = null;
			try {
				resized = DocAlignImageUtils.resizeHeatmap(heatmap, origW, origH);
				double[] located = DocAlignImageUtils.locateCorner(
					resized, config.getHeatmapThreshold());
				corners.add(new DocCorner(located[0], located[1], located[2]));
			} finally {
				heatmap.release();
				DocAlignImageUtils.releaseAll(resized);
			}
		}
		return new DocAlignResult(corners, origW, origH);
	}

	private static Mat toMat(float[][] plane) {
		if (plane.length == 0 || plane[0].length == 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"docalign 输出热力图空间维度为空");
		}
		int h = plane.length;
		int w = plane[0].length;
		float[] flat = new float[h * w];
		for (int y = 0; y < h; y++) {
			if (plane[y].length != w) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
					"docalign 输出热力图行宽不一致");
			}
			System.arraycopy(plane[y], 0, flat, y * w, w);
		}
		Mat mat = new Mat(h, w, CvType.CV_32FC1);
		mat.put(0, 0, flat);
		return mat;
	}

	private String resolveInputName() {
		Map<String, String> byName = new LinkedHashMap<>();
		try {
			for (String name : session.getSession().getInputNames()) {
				byName.put(name.toLowerCase(Locale.ROOT), name);
			}
		} catch (RuntimeException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 docalign 模型输入名失败", e);
		}
		String name = pickByName(byName, "img", "input", "images", "image", "x");
		if (name == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"docalign 模型缺少输入节点（期望 img / input / images / image / x）");
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"docalign 模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		return name;
	}

	private String resolveHeatmapOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = session.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 docalign 模型输出元信息失败", e);
		}
		if (outputInfo.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"docalign 模型应有且仅有 1 个输出，实际 " + outputInfo.size() + " 个");
		}
		String name = outputInfo.keySet().iterator().next();
		TensorInfo tensor = asTensorInfo(outputInfo.get(name));
		if (tensor == null || tensor.type != OnnxJavaType.FLOAT) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"docalign 输出应为 float32 张量，实际 " + outputInfo.get(name));
		}
		long[] shape = tensor.getShape();
		if (shape.length != IMAGE_RANK || shape[1] != EXPECTED_CORNERS) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"docalign 输出应为 [1,4,H,W]，实际 " + Arrays.toString(shape));
		}
		return name;
	}

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
					"docalign 模型 input 应为 4 维，实际 rank=" + shape.length);
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
				ErrorCode.MODEL_LOAD_FAILED, "读取 docalign 模型输入形状失败", e);
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
	 * 便于测试断言：模型输入宽度。
	 *
	 * @return 输入宽度；读取失败返回 -1
	 */
	public int modelInputWidth() {
		return modelInputSize(IMAGE_RANK - 1);
	}

	/**
	 * 便于测试断言：模型输入高度。
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
	 * 便于测试断言：模型输出节点数（DocAligner 应为 1）。
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
