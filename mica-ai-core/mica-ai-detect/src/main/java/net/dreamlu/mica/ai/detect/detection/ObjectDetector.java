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
package net.dreamlu.mica.ai.detect.detection;

import ai.onnxruntime.NodeInfo;
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
import net.dreamlu.mica.ai.detect.config.DetectConfig;
import net.dreamlu.mica.ai.detect.config.DetectModelFamily;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import net.dreamlu.mica.ai.detect.postprocess.Nms;
import net.dreamlu.mica.ai.detect.util.DetectImageUtils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * NanoDet-Plus / YOLOX-s 通用检测器（opencv_zoo 导出形态）。
 *
 * <p><b>NanoDet-Plus</b>（{@code object_detection_nanodet_2022nov.onnx}）：
 * <ul>
 *   <li>输入 {@code [1,3,416,416]}：BGR→RGB → letterbox(0, INTER_AREA) →
 *       {@code (x-mean)/std}（mean {@code [103.53,116.28,123.675]}、std {@code [57.375,57.12,58.395]}）→ CHW</li>
 *   <li>输出 <b>6 个</b> {@code [1,N,D]}：stride 8/16/32 的 cls（{@code D=numClasses}）与
 *       bbox（{@code D=4*(regMax+1)=32}）DFL 分布，按锚点数两两配对</li>
 *   <li>解码：DFL 在 8 个 bin 上做 softmax 并期望投影得到 4 条边距离，锚点为
 *       {@code (x*stride + 0.5*(stride-1))}，NMS <b>类别无关</b>（对齐 cv2.dnn.NMSBoxes）</li>
 * </ul>
 *
 * <p><b>YOLOX-s</b>（{@code object_detection_yolox_2022nov.onnx}）：
 * <ul>
 *   <li>输入 {@code [1,3,640,640]}：BGR→RGB → letterbox(114, INTER_LINEAR) → CHW，
 *       <b>不做归一化</b>（保留 0~255）</li>
 *   <li>输出 1 个 {@code [1,8400,85]}：{@code (cx,cy,w,h,obj,80 类)}，已在导出图中做过 sigmoid</li>
 *   <li>解码：{@code cx=(gx+grid)*stride}、{@code w=exp(gw)*stride}，得分 {@code obj*cls}，
 *       NMS <b>按类别</b>（对齐 cv2.dnn.NMSBoxesBatched）</li>
 * </ul>
 *
 * <p>线程安全：ONNX session 线程安全，每次推理新建 1 个 {@code OnnxTensor}；
 * 通常由 {@link net.dreamlu.mica.ai.detect.DetectEngine} 单例持有。
 */
@Slf4j
public class ObjectDetector implements AutoCloseable {

	private static final int IMAGE_RANK = 4;
	private static final int CHANNEL_DIM = 1;
	private static final int BBOX_COORDS = 4;

	private static final int[] NANODET_STRIDES = new int[]{8, 16, 32};
	private static final int[] YOLOX_STRIDES = new int[]{8, 16, 32};

	private static final float[] NANODET_MEAN = new float[]{103.53f, 116.28f, 123.675f};
	private static final float[] NANODET_STD = new float[]{57.375f, 57.12f, 58.395f};

	private final OrtEnvironment environment;
	private final OnnxModelSession session;
	@Getter
	private final DetectConfig config;
	private final String imageInputName;
	private final List<String> outputNames;

	/**
	 * 构造检测器，加载 ONNX 模型并校验输入输出契约。
	 *
	 * @param environment    ONNX 运行环境
	 * @param config         检测配置
	 * @param sessionOptions ONNX 会话选项
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载失败；
	 *                         {@link ErrorCode#ILLEGAL_ARGUMENT} 配置与模型形状不一致
	 */
	public ObjectDetector(OrtEnvironment environment, DetectConfig config,
						  OrtSession.SessionOptions sessionOptions) {
		this.environment = environment;
		this.config = config;
		this.session = new OnnxModelSession(environment, config.resolveModelPath(),
			sessionOptions, "detect");
		this.imageInputName = resolveInputName();
		this.outputNames = new ArrayList<>(session.getSession().getOutputNames());
		validateInputSize();
		log.info("mica-ai-detect 初始化完成: family={} input={} outputs={} score={} nms={}",
			config.getFamily(), imageInputName, outputNames.size(),
			config.getScoreThreshold(), config.getNmsThreshold());
	}

	/**
	 * 检测单张图像。
	 *
	 * @param bgr 原图 BGR Mat；null 或空图时返回 null
	 * @return 检测结果（原图坐标系）；输入为空时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 预处理或推理失败
	 */
	public DetectResult detect(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return null;
		}
		int origW = bgr.cols();
		int origH = bgr.rows();
		int size = config.getInputSize();
		DetectImageUtils.Letterbox letterbox = DetectImageUtils.letterbox(bgr, size,
			padValue(), interpolation());
		float[] tensorData = DetectImageUtils.toChwFloat(letterbox.image,
			mean(), std());
		OnnxTensor inputTensor = null;
		try {
			inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(tensorData),
				new long[]{1, 3, size, size});
			Map<String, OnnxTensor> inputs = new HashMap<>(1);
			inputs.put(imageInputName, inputTensor);
			try (OrtSession.Result result = session.getSession().run(inputs)) {
				List<DetectBox> candidates = config.getFamily() == DetectModelFamily.NANODET_PLUS
					? decodeNanodet(result)
					: decodeYolox(result);
				return finish(candidates, letterbox, origW, origH);
			} catch (OrtException e) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "detect 模型推理失败", e);
			}
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "创建 detect 输入张量失败", e);
		} finally {
			closeQuietly(inputTensor);
			if (letterbox.image != null) {
				letterbox.image.release();
			}
		}
	}

	/**
	 * 模型输入边长（NCHW 的最后两维）。
	 *
	 * @return 输入边长；读取失败返回 -1
	 */
	public int modelInputSize() {
		try {
			TensorInfo tensor = asTensorInfo(session.getSession().getInputInfo().get(imageInputName));
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

	/**
	 * 模型输出节点个数（NanoDet-Plus 为 6，YOLOX-s 为 1）。
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

	// ---------------------------------------------------------------- 解码

	/**
	 * NanoDet-Plus 解码：按 N（锚点数）把 8 个输出配成「cls + bbox」，再按 stride 升序解码。
	 */
	private List<DetectBox> decodeNanodet(OrtSession.Result result) throws OrtException {
		List<float[][]> values = new ArrayList<>(outputNames.size());
		List<Integer> counts = new ArrayList<>(outputNames.size());
		int binsDim = Integer.MAX_VALUE;
		int classDim = 0;
		for (int i = 0; i < outputNames.size(); i++) {
			float[][] value = asTwoDim(result.get(i).getValue());
			if (value == null) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
					"NanoDet 输出 " + outputNames.get(i) + " 不是 3 维 float 张量");
			}
			values.add(value);
			counts.add(value.length);
			int dim = value.length == 0 ? 0 : value[0].length;
			binsDim = Math.min(binsDim, dim);
			classDim = Math.max(classDim, dim);
		}
		if (outputNames.size() != NANODET_STRIDES.length * 2 || binsDim == Integer.MAX_VALUE) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"NanoDet 应输出 6 个张量（每 stride 一对 cls/bbox），实际 " + outputNames.size());
		}
		int binsPerSide = binsDim / BBOX_COORDS;
		if (binsPerSide <= 0 || binsDim % BBOX_COORDS != 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"NanoDet bbox 输出维度非法: " + binsDim);
		}

		Map<Integer, float[][]> clsByCount = new LinkedHashMap<>();
		Map<Integer, float[][]> boxByCount = new HashMap<>();
		for (int i = 0; i < values.size(); i++) {
			float[][] value = values.get(i);
			int dim = value.length == 0 ? 0 : value[0].length;
			if (dim == binsDim) {
				boxByCount.put(counts.get(i), value);
			} else if (dim == classDim) {
				clsByCount.put(counts.get(i), value);
			}
		}

		int size = config.getInputSize();
		List<DetectBox> candidates = new ArrayList<>();
		for (int stride : NANODET_STRIDES) {
			int fw = size / stride;
			int fh = size / stride;
			int count = fw * fh;
			float[][] cls = clsByCount.get(count);
			float[][] box = boxByCount.get(count);
			if (cls == null || box == null) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
					"NanoDet 缺少 stride=" + stride + " 的输出（期望锚点数 " + count + "）");
			}
			decodeNanodetLevel(cls, box, stride, fw, fh, binsPerSide, size, candidates);
		}
		return candidates;
	}

	private void decodeNanodetLevel(float[][] cls, float[][] box, int stride, int fw, int fh,
									int binsPerSide, int size, List<DetectBox> out) {
		int index = 0;
		for (int y = 0; y < fh; y++) {
			for (int x = 0; x < fw; x++, index++) {
				float cx = x * stride + 0.5f * (stride - 1);
				float cy = y * stride + 0.5f * (stride - 1);
				float[] dist = dflDistances(box[index], binsPerSide, stride);

				float x1 = clamp(cx - dist[0], 0f, size);
				float y1 = clamp(cy - dist[1], 0f, size);
				float x2 = clamp(cx + dist[2], 0f, size);
				float y2 = clamp(cy + dist[3], 0f, size);

				int bestClass = -1;
				float bestScore = 0f;
				float[] scores = cls[index];
				for (int c = 0; c < scores.length; c++) {
					if (scores[c] > bestScore) {
						bestScore = scores[c];
						bestClass = c;
					}
				}
				if (bestClass < 0 || bestScore < config.getScoreThreshold()) {
					continue;
				}
				out.add(new DetectBox(x1, y1, x2, y2, bestScore, bestClass));
			}
		}
	}

	/**
	 * DFL（Distribution Focal Loss）解码：对每条边的 {@code binsPerSide} 个 bin 做 softmax，
	 * 与 {@code 0..binsPerSide-1} 求期望，再乘以 stride 得到像素距离。
	 */
	private static float[] dflDistances(float[] raw, int binsPerSide, int stride) {
		float[] dist = new float[BBOX_COORDS];
		for (int side = 0; side < BBOX_COORDS; side++) {
			int offset = side * binsPerSide;
			float max = Float.NEGATIVE_INFINITY;
			for (int k = 0; k < binsPerSide; k++) {
				max = Math.max(max, raw[offset + k]);
			}
			float sum = 0f;
			float expect = 0f;
			for (int k = 0; k < binsPerSide; k++) {
				float e = (float) Math.exp(raw[offset + k] - max);
				sum += e;
				expect += e * k;
			}
			dist[side] = sum <= 0f ? 0f : expect / sum * stride;
		}
		return dist;
	}

	/**
	 * YOLOX-s 解码：单一 {@code [1,N,85]} 输出，按 stride 8/16/32 顺序平铺。
	 */
	private List<DetectBox> decodeYolox(OrtSession.Result result) throws OrtException {
		if (outputNames.size() != 1) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"YOLOX 应只输出 1 个张量，实际 " + outputNames.size());
		}
		float[][] dets = asTwoDim(result.get(0).getValue());
		if (dets == null) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "YOLOX 输出不是 3 维 float 张量");
		}
		int size = config.getInputSize();
		int numClasses = dets.length == 0 ? 0 : dets[0].length - 5;
		if (numClasses <= 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"YOLOX 输出维度非法: " + (dets.length == 0 ? 0 : dets[0].length));
		}

		List<DetectBox> candidates = new ArrayList<>();
		int index = 0;
		for (int stride : YOLOX_STRIDES) {
			int f = size / stride;
			for (int y = 0; y < f; y++) {
				for (int x = 0; x < f; x++, index++) {
					if (index >= dets.length) {
						throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
							"YOLOX 输出行数不足，期望至少 " + index + " 行");
					}
					float[] row = dets[index];
					float cx = (row[0] + x) * stride;
					float cy = (row[1] + y) * stride;
					float w = (float) Math.exp(row[2]) * stride;
					float h = (float) Math.exp(row[3]) * stride;
					float obj = row[4];

					int bestClass = -1;
					float bestScore = 0f;
					for (int c = 0; c < numClasses; c++) {
						float score = row[5 + c];
						if (score > bestScore) {
							bestScore = score;
							bestClass = c;
						}
					}
					float score = obj * bestScore;
					if (bestClass < 0 || score < config.getScoreThreshold()) {
						continue;
					}
					DetectBox mapped = new DetectBox(
						clamp(cx - w / 2f, 0f, size), clamp(cy - h / 2f, 0f, size),
						clamp(cx + w / 2f, 0f, size), clamp(cy + h / 2f, 0f, size),
						score, bestClass);
					if (mapped.width() > 0f && mapped.height() > 0f) {
						candidates.add(mapped);
					}
				}
			}
		}
		return candidates;
	}

	/**
	 * NMS → 反算回原图坐标 → 截断。
	 */
	private DetectResult finish(List<DetectBox> candidates, DetectImageUtils.Letterbox letterbox,
								int origW, int origH) {
		boolean classAgnostic = config.getFamily() == DetectModelFamily.NANODET_PLUS;
		List<DetectBox> kept = Nms.suppress(candidates, config.getNmsThreshold(),
			classAgnostic, config.getMaxDetections());

		List<DetectBox> mapped = new ArrayList<>(kept.size());
		for (DetectBox box : kept) {
			float x1 = clamp((box.getX1() - letterbox.left) / letterbox.ratio, 0f, origW);
			float y1 = clamp((box.getY1() - letterbox.top) / letterbox.ratio, 0f, origH);
			float x2 = clamp((box.getX2() - letterbox.left) / letterbox.ratio, 0f, origW);
			float y2 = clamp((box.getY2() - letterbox.top) / letterbox.ratio, 0f, origH);
			mapped.add(new DetectBox(x1, y1, x2, y2, box.getScore(), box.getClassId()));
		}
		return new DetectResult(origW, origH, mapped);
	}

	// ---------------------------------------------------------------- 契约

	private int padValue() {
		return config.getFamily() == DetectModelFamily.YOLOX_S ? 114 : 0;
	}

	private int interpolation() {
		return config.getFamily() == DetectModelFamily.YOLOX_S
			? Imgproc.INTER_LINEAR : Imgproc.INTER_AREA;
	}

	private float[] mean() {
		return config.getFamily() == DetectModelFamily.NANODET_PLUS ? NANODET_MEAN : null;
	}

	private float[] std() {
		return config.getFamily() == DetectModelFamily.NANODET_PLUS ? NANODET_STD : null;
	}

	private String resolveInputName() {
		Map<String, String> byName = new LinkedHashMap<>();
		try {
			for (String name : session.getSession().getInputNames()) {
				byName.put(name.toLowerCase(Locale.ROOT), name);
			}
		} catch (RuntimeException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "读取 detect 模型输入名失败", e);
		}
		if (byName.size() != 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"detect 模型应有且仅有 1 个输入，实际 " + byName.size() + " 个: " + byName.values());
		}
		return byName.values().iterator().next();
	}

	private void validateInputSize() {
		try {
			NodeInfo nodeInfo = session.getSession().getInputInfo().get(imageInputName);
			TensorInfo tensor = asTensorInfo(nodeInfo);
			if (tensor == null) {
				return;
			}
			long[] shape = tensor.getShape();
			if (shape.length != IMAGE_RANK) {
				throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
					"detect 模型输入应为 4 维，实际 rank=" + shape.length);
			}
			long modelH = shape[IMAGE_RANK - 2];
			long modelW = shape[IMAGE_RANK - 1];
			if (shape[CHANNEL_DIM] != 3 && shape[CHANNEL_DIM] > 0) {
				throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
					"detect 模型输入应为 3 通道，实际 " + shape[CHANNEL_DIM]);
			}
			if (modelH > 0 && modelW > 0
				&& (modelH != config.getInputSize() || modelW != config.getInputSize())) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"inputSize 必须等于模型输入边长 " + modelH + "，当前为 " + config.getInputSize());
			}
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "读取 detect 模型输入形状失败", e);
		}
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
		return null;
	}

	private static float clamp(float v, float min, float max) {
		return v < min ? min : (v > max ? max : v);
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
	 * 便于测试断言：模型输出节点名列表（只读）。
	 *
	 * @return 输出节点名，读取失败时返回空列表
	 */
	public List<String> getOutputNames() {
		return Collections.unmodifiableList(outputNames);
	}
}
