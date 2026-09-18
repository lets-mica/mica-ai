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
package net.dreamlu.mica.ai.face.detection;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * 人脸检测器，基于 <b>YuNet</b> ONNX 模型（OpenCV Zoo {@code face_detection_yunet_2023mar.onnx}）。
 *
 * <p>模型规格（已用 OpenCV {@code cv::FaceDetectorYN} 逐位比对确认）：
 * <ul>
 *     <li><b>输入</b>：{@code [1, 3, 640, 640]} float32，<b>BGR 顺序、原始 0~255 像素值</b>。
 *         不做均值扣除、不做 /255 缩放（图内亦无归一化算子）。</li>
 *     <li><b>输出</b>：固定 12 个，顺序为
 *         {@code cls_8, cls_16, cls_32, obj_8, obj_16, obj_32,
 *         bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32}，
 *         形状分别为 {@code [1, N, 1]}、{@code [1, N, 4]}、{@code [1, N, 10]}。</li>
 *     <li>cls 与 obj 在导出时<b>已过 sigmoid</b>，分值取 {@code sqrt(cls * obj)}。</li>
 *     <li>框解码为 anchor-free：{@code cx = (c + dx) * stride}，{@code w = exp(dw) * stride}。</li>
 * </ul>
 *
 * <p>线程安全：ONNX Runtime OrtSession 本身线程安全；本类为 stateless，可以作为单例 Bean
 * 在 Spring 容器中共享；唯一的可变状态是每次推理新建的 {@code OnnxTensor}，try-with-resources
 * 保证释放，无跨线程共享问题。
 */
@Getter
public class FaceDetector {

	public static final int INPUT_SIZE = 640;
	private static final int MIN_TILE_SIDE = 48;
	private static final int[] STRIDES = {8, 16, 32};
	private static final float SCALE = 1.0f;
	private static final float OFFSET = 0.0f;

	private final OrtSession session;
	private final OrtEnvironment env;
	private final float threshold;
	private final float nmsThreshold;

	public FaceDetector(ModelManager modelManager) {
		this.session = modelManager.getDetectionSession();
		this.env = modelManager.getEnvironment();
		this.threshold = modelManager.getConfig().getDetectionThreshold();
		this.nmsThreshold = modelManager.getConfig().getNmsThreshold();
	}

	private static float clip(float v, float lo, float hi) {
		return Math.max(lo, Math.min(hi, v));
	}

	private static float iou(FaceBox a, FaceBox b) {
		float ax1 = (float) Math.floor(a.getX1());
		float ay1 = (float) Math.floor(a.getY1());
		float ax2 = (float) Math.floor(a.getX2());
		float ay2 = (float) Math.floor(a.getY2());
		float bx1 = (float) Math.floor(b.getX1());
		float by1 = (float) Math.floor(b.getY1());
		float bx2 = (float) Math.floor(b.getX2());
		float by2 = (float) Math.floor(b.getY2());

		float interX1 = Math.max(ax1, bx1);
		float interY1 = Math.max(ay1, by1);
		float interX2 = Math.min(ax2, bx2);
		float interY2 = Math.min(ay2, by2);
		float iw = interX2 - interX1;
		float ih = interY2 - interY1;
		if (iw <= 0 || ih <= 0) {
			return 0f;
		}
		float inter = iw * ih;
		float union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter;
		return union < 1e-10f ? 0f : inter / union;
	}

	public static List<FaceBox> nms(List<FaceBox> boxes, float iouThreshold) {
		if (boxes == null || boxes.isEmpty()) {
			return Collections.emptyList();
		}
		List<FaceBox> sorted = new ArrayList<>(boxes);
		sorted.sort((a, b) -> Float.compare(b.getScore(), a.getScore()));
		List<FaceBox> result = new ArrayList<>();
		boolean[] suppressed = new boolean[sorted.size()];
		for (int i = 0; i < sorted.size(); i++) {
			if (suppressed[i]) {
				continue;
			}
			FaceBox keep = sorted.get(i);
			result.add(keep);
			for (int j = i + 1; j < sorted.size(); j++) {
				if (!suppressed[j] && iou(keep, sorted.get(j)) > iouThreshold) {
					suppressed[j] = true;
				}
			}
		}
		return result;
	}

	public List<FaceBox> detect(Mat image) {
		return detect(image, threshold);
	}

	public List<FaceBox> detect(Mat image, float threshold) {
		if (image == null || image.empty()) {
			return Collections.emptyList();
		}
		int origW = image.cols();
		int origH = image.rows();

		float scale = Math.min((float) INPUT_SIZE / origW, (float) INPUT_SIZE / origH);
		int newW = Math.max(1, Math.round(origW * scale));
		int newH = Math.max(1, Math.round(origH * scale));
		float invScale = 1f / scale;

		Mat resized = new Mat();
		Mat canvas = new Mat();
		try {
			Imgproc.resize(image, resized, new Size(newW, newH), 0, 0, Imgproc.INTER_LINEAR);
			Core.copyMakeBorder(resized, canvas, 0, INPUT_SIZE - newH, 0, INPUT_SIZE - newW,
				Core.BORDER_CONSTANT, new Scalar(0, 0, 0));

			float[] data = ImageUtils.bgrHwcToChwFloat(canvas, SCALE, OFFSET);
			try (OnnxTensor inputTensor = createInput(data)) {
				String inputName = session.getInputNames().iterator().next();
				Map<String, OnnxTensor> inputs = new HashMap<>();
				inputs.put(inputName, inputTensor);
				try (OrtSession.Result result = session.run(inputs)) {
					return decodeAndNms(result, invScale, origW, origH, threshold);
				} catch (OrtException e) {
					throw new MicaAiException(
						ErrorCode.DETECTION_FAILED, "YuNet 推理失败", e);
				}
			}
		} finally {
			resized.release();
			canvas.release();
		}
	}

	public List<FaceBox> detectTiled(Mat image, int tileSize, double overlap, float threshold) {
		if (image == null || image.empty()) {
			return Collections.emptyList();
		}
		int w = image.cols();
		int h = image.rows();
		if (w <= tileSize && h <= tileSize) {
			return detect(image, threshold);
		}
		int step = Math.max(1, (int) Math.round(tileSize * (1.0 - overlap)));
		List<FaceBox> all = new ArrayList<>();
		for (int y : tileStarts(h, tileSize, step)) {
			for (int x : tileStarts(w, tileSize, step)) {
				int tw = Math.min(tileSize, w - x);
				int th = Math.min(tileSize, h - y);
				if (tw < MIN_TILE_SIDE || th < MIN_TILE_SIDE) {
					continue;
				}
				Mat tile = new Mat(image, new Rect(x, y, tw, th));
				try {
					for (FaceBox b : detect(tile, threshold)) {
						all.add(shift(b, x, y));
					}
				} finally {
					tile.release();
				}
			}
		}
		return nms(all, nmsThreshold);
	}

	static int[] tileStarts(int length, int tile, int step) {
		if (length <= tile) {
			return new int[]{0};
		}
		LinkedHashSet<Integer> starts = new LinkedHashSet<>();
		for (int s = 0; s + tile <= length; s += step) {
			starts.add(s);
		}
		starts.add(length - tile);
		int[] out = new int[starts.size()];
		int i = 0;
		for (int v : starts) {
			out[i++] = v;
		}
		return out;
	}

	private static FaceBox shift(FaceBox box, int dx, int dy) {
		float[][] src = box.getLandmarks();
		float[][] lm = null;
		if (src != null) {
			lm = new float[src.length][2];
			for (int i = 0; i < src.length; i++) {
				lm[i][0] = src[i][0] + dx;
				lm[i][1] = src[i][1] + dy;
			}
		}
		return new FaceBox(box.getX1() + dx, box.getY1() + dy,
			box.getX2() + dx, box.getY2() + dy, box.getScore(), lm);
	}

	private List<FaceBox> decodeAndNms(OrtSession.Result result, float invScale,
									   int origW, int origH, float threshold) {
		final int headCount = 4;
		final int strideCount = STRIDES.length;
		if (result.size() != headCount * strideCount) {
			throw new MicaAiException(
				ErrorCode.DETECTION_FAILED,
				"YuNet 输出数量异常，期望 " + (headCount * strideCount) + "，实际 " + result.size());
		}

		List<FaceBox> candidates = new ArrayList<>();
		for (int s = 0; s < strideCount; s++) {
			int stride = STRIDES[s];
			int featSize = INPUT_SIZE / stride;
			int featArea = featSize * featSize;

			float[][] cls = readFlat(result, 0 * strideCount + s, 1, featArea, stride);
			float[][] obj = readFlat(result, 1 * strideCount + s, 1, featArea, stride);
			float[][] bbox = readFlat(result, 2 * strideCount + s, 4, featArea, stride);
			float[][] kps = readFlat(result, 3 * strideCount + s, 10, featArea, stride);

			for (int idx = 0; idx < featArea; idx++) {
				int c = idx % featSize;
				int r = idx / featSize;

				float clsScore = clip(cls[idx][0], 0f, 1f);
				float objScore = clip(obj[idx][0], 0f, 1f);
				float score = (float) Math.sqrt((double) clsScore * objScore);
				if (score < threshold) {
					continue;
				}

				float cx = (c + bbox[idx][0]) * stride;
				float cy = (r + bbox[idx][1]) * stride;
				float w = (float) Math.exp(bbox[idx][2]) * stride;
				float h = (float) Math.exp(bbox[idx][3]) * stride;

				float x1 = cx - w / 2f;
				float y1 = cy - h / 2f;
				float x2 = cx + w / 2f;
				float y2 = cy + h / 2f;

				float[][] landmarks = new float[5][2];
				boolean valid = true;
				for (int k = 0; k < 5; k++) {
					landmarks[k][0] = (kps[idx][k * 2] + c) * stride;
					landmarks[k][1] = (kps[idx][k * 2 + 1] + r) * stride;
					if (!Float.isFinite(landmarks[k][0]) || !Float.isFinite(landmarks[k][1])) {
						valid = false;
						break;
					}
				}
				if (!valid) {
					continue;
				}

				float ox1 = clip(x1 * invScale, 0, origW);
				float oy1 = clip(y1 * invScale, 0, origH);
				float ox2 = clip(x2 * invScale, 0, origW);
				float oy2 = clip(y2 * invScale, 0, origH);
				float[][] oKps = new float[5][2];
				for (int k = 0; k < 5; k++) {
					oKps[k][0] = clip(landmarks[k][0] * invScale, 0, origW);
					oKps[k][1] = clip(landmarks[k][1] * invScale, 0, origH);
				}

				if (ox2 - ox1 < 1f || oy2 - oy1 < 1f) {
					continue;
				}
				candidates.add(new FaceBox(ox1, oy1, ox2, oy2, score, oKps));
			}
		}
		return nms(candidates, nmsThreshold);
	}

	private float[][] readFlat(OrtSession.Result result, int index,
							   int channels, int featArea, int stride) {
		OnnxValue value = result.get(index);
		if (!(value instanceof OnnxTensor)) {
			throw new MicaAiException(
				ErrorCode.DETECTION_FAILED,
				"YuNet 第 " + index + " 个输出不是 tensor: "
					+ (value == null ? "null" : value.getClass().getName()));
		}
		OnnxTensor tensor = (OnnxTensor) value;
		try {
			float[][][] arr = (float[][][]) tensor.getValue();
			if (arr.length != 1) {
				throw new MicaAiException(
					ErrorCode.DETECTION_FAILED,
					"YuNet 第 " + index + " 个输出 batch 维应为 1，实际 " + arr.length);
			}
			float[][] rows = arr[0];
			if (rows.length != featArea || (rows.length > 0 && rows[0].length != channels)) {
				throw new MicaAiException(
					ErrorCode.DETECTION_FAILED,
					"YuNet 第 " + index + " 个输出形状不符合 stride " + stride
						+ " 预期：期望 [" + featArea + ", " + channels
						+ "]，实际 [" + rows.length + ", "
						+ (rows.length == 0 ? 0 : rows[0].length) + "]");
			}
			return rows;
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.DETECTION_FAILED, "读取 YuNet 输出失败", e);
		}
	}

	private OnnxTensor createInput(float[] data) {
		try {
			return OnnxTensor.createTensor(env, FloatBuffer.wrap(data),
				new long[]{1, 3, INPUT_SIZE, INPUT_SIZE});
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.DETECTION_FAILED, "构造 YuNet 输入张量失败", e);
		}
	}
}