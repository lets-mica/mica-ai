/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.layout.detection;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.layout.config.LayoutConfig;
import net.dreamlu.mica.ai.layout.model.LayoutResult;
import net.dreamlu.mica.ai.layout.postprocess.LayoutPostProcessor;
import net.dreamlu.mica.ai.layout.util.LayoutImageUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * PP-DocLayoutV3 版面检测器（PaddleX 导出形态，I/O 实测于 2026-09-17）。
 *
 * <p>ONNX 输入：
 * <ul>
 *   <li>{@code image}：{@code [N, 3, 800, 800]} float32，BGR→RGB + CHW + {@code (x/255-mean)/std}</li>
 *   <li>{@code im_shape}：{@code [N, 2]} float32 = {@code [orig_h, orig_w]}（<b>2 维，不是 3 维</b>）</li>
 *   <li>{@code scale_factor}：{@code [N, 2]} float32；模型按
 *       {@code 输出坐标 = 原图坐标 / scale_factor} 计算，故喂 {@code 1 / letterboxScale}
 *       时输出即 letterbox 画布坐标</li>
 * </ul>
 * ONNX 输出（3 个）：
 * <ul>
 *   <li>{@code fetch_name_0}：{@code [300, 7]} float32 =
 *       {@code [cls, score, x1, y1, x2, y2, order]}，<b>未 NMS</b></li>
 *   <li>{@code fetch_name_1}：{@code [1]} int32，有效检测数（实测恒为 300）</li>
 *   <li>{@code fetch_name_2}：{@code [300, 200, 200]} int32，指针网络原始输出 —— 阅读顺序已由
 *       {@code fetch_name_0} 第 7 列给出，本检测器不消费该节点</li>
 * </ul>
 *
 * <p>输入节点名按「名称提示」解析，输出节点名按「名称提示 → 形状/dtype 推断」解析，
 * 换导出、换文件名都不会静默错位；模型输入边长与 {@link LayoutConfig#getMaxSideLength()}
 * 不一致时快速失败。
 */
@Slf4j
public class LayoutDetector implements AutoCloseable {

	private static final int MIN_BOX_COLUMNS = 6;
	private static final int IMAGE_RANK = 4;

	private final OrtEnvironment environment;
	private final OnnxModelSession session;
	@Getter
	private final LayoutConfig config;
	private final LayoutPostProcessor postProcessor;
	private final Object inferLock = new Object();
	private final String imageInputName;
	private final String imShapeInputName;
	private final String scaleInputName;
	private final String boxesOutputName;

	public LayoutDetector(OrtEnvironment environment, LayoutConfig config,
						OrtSession.SessionOptions sessionOptions) {
		this.environment = environment;
		this.config = config;
		this.session = new OnnxModelSession(
			environment, config.resolveModelPath(),
			sessionOptions, "layout-detector");
		this.postProcessor = new LayoutPostProcessor(config);
		Map<String, String> inputMeta = probeInputMeta();
		this.imageInputName = requireByName(inputMeta, "image", "input", "images");
		this.imShapeInputName = requireByName(inputMeta, "im_shape", "im_info", "orig_shape");
		this.scaleInputName = requireByName(inputMeta, "scale_factor", "scale", "scales");
		this.boxesOutputName = resolveBoxesOutputName();
		validateImageInputSize();
		log.info("mica-ai-layout 初始化完成: version={} image={} im_shape={} scale={} boxes={}",
			config.getModelVersion(), imageInputName, imShapeInputName,
			scaleInputName, boxesOutputName);
	}

	public List<LayoutResult> detect(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return LayoutResult.emptyList();
		}
		int origW = bgr.cols();
		int origH = bgr.rows();
		LetterBoxResult pre = letterBox(bgr, config.getMaxSideLength(),
			config.getMean(), config.getStd());
		try {
			synchronized (inferLock) {
				OnnxTensor imageTensor = null;
				OnnxTensor imShapeTensor = null;
				OnnxTensor scaleTensor = null;
				Map<String, OnnxTensor> inputs = new HashMap<>(3);
				try {
					imageTensor = OnnxTensor.createTensor(environment,
						FloatBuffer.wrap(pre.tensorData),
						new long[]{1, 3, pre.side, pre.side});
					imShapeTensor = OnnxTensor.createTensor(environment,
						FloatBuffer.wrap(new float[]{(float) origH, (float) origW}),
						new long[]{1, 2});
					float invScale = (float) (1d / pre.scale);
					scaleTensor = OnnxTensor.createTensor(environment,
						FloatBuffer.wrap(new float[]{invScale, invScale}),
						new long[]{1, 2});
					inputs.put(imageInputName, imageTensor);
					inputs.put(imShapeInputName, imShapeTensor);
					inputs.put(scaleInputName, scaleTensor);
					try (OrtSession.Result result = session.getSession().run(inputs)) {
						return parseAndPostProcess(result, pre, origW, origH);
					} catch (OrtException e) {
						throw new MicaAiException(
							ErrorCode.INFERENCE_FAILED, "layout 模型推理失败", e);
					}
				} finally {
					closeQuietly(imageTensor);
					closeQuietly(imShapeTensor);
					closeQuietly(scaleTensor);
				}
			}
		} catch (Exception e) {
			if (e instanceof MicaAiException) {
				throw (MicaAiException) e;
			}
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "layout 推理异常", e);
		}
	}

	public List<LayoutResult> detectBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = LayoutImageUtils.byteArrayToMat(imageBytes);
			return detect(bgr);
		} finally {
			LayoutImageUtils.releaseAll(bgr);
		}
	}

	static LetterBoxResult letterBox(Mat bgr, int maxSide, float[] mean, float[] std) {
		int h = bgr.rows();
		int w = bgr.cols();
		double scale = Math.min(maxSide / (double) h, maxSide / (double) w);
		int newH = Math.max(1, (int) Math.round(h * scale));
		int newW = Math.max(1, (int) Math.round(w * scale));

		Mat resized = new Mat();
		Mat padded = new Mat();
		try {
			Imgproc.resize(bgr, resized, new Size(newW, newH));
			Core.copyMakeBorder(resized, padded, 0, maxSide - newH, 0, maxSide - newW,
				Core.BORDER_CONSTANT, new Scalar(114, 114, 114));
			// 注意：bgrHwcToRgbChwFloat 自身完成 BGR→RGB，此处不可先 cvtColor，
			// 否则会「BGR→RGB→按 BGR 读回」等于把 BGR 喂给模型（实测该图过阈值框数 1 → 0）。
			float[] data = LayoutImageUtils.bgrHwcToRgbChwFloat(padded, mean, std);
			return new LetterBoxResult(data, scale, maxSide);
		} finally {
			resized.release();
			padded.release();
		}
	}

	private Map<String, String> probeInputMeta() {
		Map<String, String> map = new HashMap<>();
		try {
			for (String name : session.getSession().getInputNames()) {
				map.put(name.toLowerCase(Locale.ROOT), name);
			}
		} catch (RuntimeException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 layout 模型输入元信息失败", e);
		}
		return map;
	}

	private static String requireByName(Map<String, String> meta, String... hints) {
		String name = pickByName(meta, hints);
		if (name == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"layout 模型缺少输入节点: " + String.join("/", hints));
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

	/**
	 * 定位检测输出节点：优先名称提示（boxes/bbox），否则取「float32 且最后一维 &ge; 6 的 2 维输出」中
	 * 列数最大者 —— PP-DocLayoutV3 导出的节点名是 {@code fetch_name_0}，只能靠形状识别。
	 */
	private String resolveBoxesOutputName() {
		Map<String, NodeInfo> outputInfo;
		try {
			outputInfo = session.getSession().getOutputInfo();
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 layout 模型输出元信息失败", e);
		}
		Map<String, String> byName = new HashMap<>();
		for (String name : outputInfo.keySet()) {
			byName.put(name.toLowerCase(Locale.ROOT), name);
		}
		String hinted = pickByName(byName, "boxes", "bbox", "detection_boxes");
		if (hinted != null) {
			return hinted;
		}
		String best = null;
		long bestColumns = 0;
		for (Map.Entry<String, NodeInfo> entry : outputInfo.entrySet()) {
			TensorInfo tensor = asTensorInfo(entry.getValue());
			if (tensor == null || tensor.type != OnnxJavaType.FLOAT) {
				continue;
			}
			long[] shape = tensor.getShape();
			if (shape.length != 2 || shape[1] < MIN_BOX_COLUMNS) {
				continue;
			}
			if (shape[1] > bestColumns) {
				bestColumns = shape[1];
				best = entry.getKey();
			}
		}
		if (best == null) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"未能在 layout 模型中定位检测输出节点（期望 float32 [N, >=6]）");
		}
		return best;
	}

	/**
	 * 校验模型输入边长与 {@code maxSideLength} 一致，否则 letterbox 结果与模型张量形状对不上。
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
					"layout 模型 image 输入应为 4 维，实际 rank=" + shape.length);
			}
			long modelH = shape[IMAGE_RANK - 2];
			long modelW = shape[IMAGE_RANK - 1];
			if (modelH > 0 && modelW > 0
				&& (modelH != config.getMaxSideLength() || modelW != config.getMaxSideLength())) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"maxSideLength 必须等于模型输入边长 " + modelH + "，当前为 "
						+ config.getMaxSideLength());
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "读取 layout 模型输入形状失败", e);
		}
	}

	private static TensorInfo asTensorInfo(NodeInfo nodeInfo) {
		if (nodeInfo == null) {
			return null;
		}
		ValueInfo info = nodeInfo.getInfo();
		return info instanceof TensorInfo ? (TensorInfo) info : null;
	}

	private List<LayoutResult> parseAndPostProcess(OrtSession.Result result,
												   LetterBoxResult pre,
												   int origW, int origH) throws OrtException {
		Object boxValue = result.get(boxesOutputName).get().getValue();
		float[][] boxes = asTwoDim(boxValue);
		if (boxes == null) {
			log.warn("layout 检测输出 {} 不是 2 维 float 张量，返回空结果", boxesOutputName);
			return LayoutResult.emptyList();
		}
		return postProcessor.postProcess(boxes, pre.scale, 0, 0, origW, origH);
	}

	private static float[][] asTwoDim(Object value) {
		if (value instanceof float[][]) {
			return (float[][]) value;
		}
		if (value instanceof float[][][]) {
			float[][][] three = (float[][][]) value;
			return three.length > 0 ? three[0] : null;
		}
		if (value instanceof float[]) {
			return new float[][]{((float[]) value)};
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

	@Override
	public void close() {
		if (session != null) {
			session.close();
		}
	}

	@Getter
	@AllArgsConstructor
	static class LetterBoxResult {
		final float[] tensorData;
		final double scale;
		/**
		 * 补边后的方形画布边长（= 模型输入边长），张量形状必须用它而不是缩放后的内容尺寸
		 */
		final int side;
	}
}
