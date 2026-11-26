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
package net.dreamlu.mica.ai.detect.core;

import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.detect.config.DetectConfig;
import net.dreamlu.mica.ai.detect.config.DetectModelFamily;
import net.dreamlu.mica.ai.detect.config.SlicedDetectConfig;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import net.dreamlu.mica.ai.detect.model.SliceMergeStrategy;
import net.dreamlu.mica.ai.detect.postprocess.Nms;
import net.dreamlu.mica.ai.detect.sliced.Slice;
import net.dreamlu.mica.ai.detect.sliced.SliceGrid;
import net.dreamlu.mica.ai.detect.util.DetectImageUtils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import java.util.ArrayList;
import java.util.List;

/**
 * 切片推理门面：大图切成多片分别推理，再合并成一份原图坐标系的检测结果。
 *
 * <p><b>解决什么问题</b>：整图推理时 letterbox 会把整张图缩到模型输入边长，
 * 一个 30×30 的远处目标在 2000×1500 的图上只剩约 8px，模型根本看不见；
 * 切片后该目标按原像素推理，清晰可见。这是小目标召回的主要手段。
 *
 * <p><b>切片边长恒等于模型输入边长</b>（{@link DetectConfig#getInputSize()}）：
 * 只有这样 letterbox 的缩放比才恒为 1、无填充，每片与整图推理的预处理契约完全一致。
 * 图像边缘不足一整片的切片会被补齐到同样大小（填充值与模型族一致），
 * 避免被 letterbox 放大导致小目标优势归零。
 *
 * <p><b>不持有模型</b>：底层 {@link DetectEngine} 由调用方持有与释放，
 * 本类仅做切片编排，因此不实现 {@code AutoCloseable}。
 *
 * <p>线程安全：无内部可变状态，可作为单例共享。
 *
 * <pre>{@code
 * try (DetectEngine engine = DetectEngine.create(DetectConfig.builder()
 *         .modelPath("model-tools/detect/models/object_detection_nanodet_2022nov.onnx")
 *         .build())) {
 *     SlicedDetectEngine sliced = SlicedDetectEngine.create(engine,
 *         SlicedDetectConfig.builder()
 *             .detectConfig(engine.getConfig())
 *             .overlapRatio(0.25f)
 *             .build());
 *     DetectResult result = sliced.detect(bgr);
 * }
 * }</pre>
 */
public class SlicedDetectEngine {

	/**
	 * 底层检测能力（调用方持有，本类不负责关闭）。
	 */
	@Getter
	private final DetectSource source;

	/**
	 * 切片配置。
	 */
	@Getter
	private final SlicedDetectConfig config;

	/**
	 * 构造切片推理引擎。
	 *
	 * @param source 底层检测能力
	 * @param config 切片配置
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 配置非法
	 */
	public SlicedDetectEngine(DetectSource source, SlicedDetectConfig config) {
		if (source == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "检测能力不能为空");
		}
		if (config == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "SlicedDetectConfig 不能为空");
		}
		config.validate();
		this.source = source;
		this.config = config;
	}

	/**
	 * 创建切片推理引擎。
	 *
	 * @param source 底层检测能力（生产环境传 {@link DetectEngine}）
	 * @param config 切片配置
	 * @return 切片推理引擎
	 */
	public static SlicedDetectEngine create(DetectSource source, SlicedDetectConfig config) {
		return new SlicedDetectEngine(source, config);
	}

	/**
	 * 切片检测单张图像。
	 *
	 * @param bgr 原图 BGR Mat；null 或空图时返回 null
	 * @return 检测结果（原图坐标系）；输入为空时返回 null
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 切片数超过上限；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public DetectResult detect(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return null;
		}
		int width = bgr.cols();
		int height = bgr.rows();
		int size = config.resolveSliceSize();
		int step = config.resolveStep();
		SliceGrid grid = SliceGrid.of(width, height, size, step);
		int sliceCount = grid.getSliceCount();
		if (sliceCount > config.getMaxSlices()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"图像 " + width + "x" + height + " 在 " + size + " 切片（重叠 "
					+ (size - step) + "px）下需要 " + sliceCount + " 片，超过上限 "
					+ config.getMaxSlices() + " 片；请调大 max-slices 或改用更大输入边长的模型");
		}

		List<DetectBox> collected = new ArrayList<>();
		for (Slice slice : grid.getSlices()) {
			Mat tile = extractTile(bgr, slice, size);
			try {
				DetectResult sliceResult = source.detect(tile);
				for (DetectBox box : sliceResult.getBoxes()) {
					DetectBox mapped = mapToOriginal(box, slice, width, height);
					if (mapped != null) {
						collected.add(mapped);
					}
				}
			} finally {
				tile.release();
			}
		}
		return new DetectResult(width, height, merge(collected));
	}

	/**
	 * 切片检测图像字节。
	 *
	 * @param imageBytes 图像字节（png / jpg 等）
	 * @return 检测结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public DetectResult detectBytes(byte[] imageBytes) {
		Mat bgr = DetectImageUtils.byteArrayToMat(imageBytes);
		try {
			return detect(bgr);
		} finally {
			bgr.release();
		}
	}

	/**
	 * 计算给定尺寸会被切成几片（纯计算，不推理）。
	 *
	 * @param width  原图宽（像素）
	 * @param height 原图高（像素）
	 * @return 切片数
	 */
	public int sliceCount(int width, int height) {
		return SliceGrid.of(width, height, config.resolveSliceSize(), config.resolveStep()).getSliceCount();
	}

	// ---------------------------------------------------------------- 内部

	/**
	 * 取出切片；超出原图的部分按模型族的填充值补齐，保证切片始终是「边长 × 边长」。
	 */
	private Mat extractTile(Mat bgr, Slice slice, int size) {
		int padValue = config.getDetectConfig().getFamily().getPadValue();
		int x1 = Math.max(0, slice.getX());
		int y1 = Math.max(0, slice.getY());
		int x2 = Math.min(bgr.cols(), slice.getX2());
		int y2 = Math.min(bgr.rows(), slice.getY2());
		Scalar scalar = new Scalar(padValue, padValue, padValue);
		if (x2 <= x1 || y2 <= y1) {
			return new Mat(size, size, CvType.CV_8UC3, scalar);
		}
		Mat roi = new Mat(bgr, new Rect(x1, y1, x2 - x1, y2 - y1));
		Mat padded = new Mat();
		try {
			Core.copyMakeBorder(roi, padded, y1 - slice.getY(), slice.getY2() - y2,
				x1 - slice.getX(), slice.getX2() - x2, Core.BORDER_CONSTANT, scalar);
			return padded;
		} finally {
			roi.release();
		}
	}

	/**
	 * 把切片局部坐标的框平移到原图坐标系，并裁剪到原图边界。
	 *
	 * @return 完全落在原图外时返回 null
	 */
	private static DetectBox mapToOriginal(DetectBox box, Slice slice, int width, int height) {
		float x1 = clamp(box.getX1() + slice.getX(), 0f, width);
		float y1 = clamp(box.getY1() + slice.getY(), 0f, height);
		float x2 = clamp(box.getX2() + slice.getX(), 0f, width);
		float y2 = clamp(box.getY2() + slice.getY(), 0f, height);
		if (x2 <= x1 || y2 <= y1) {
			return null;
		}
		return new DetectBox(x1, y1, x2, y2, box.getScore(), box.getClassId());
	}

	/**
	 * 跨片合并：复用整图的 NMS 语义（类别互斥规则随模型族）。
	 */
	private List<DetectBox> merge(List<DetectBox> boxes) {
		if (config.getMergeStrategy() == SliceMergeStrategy.KEEP_ALL) {
			return boxes;
		}
		DetectConfig detectConfig = config.getDetectConfig();
		boolean classAgnostic = detectConfig.getFamily() == DetectModelFamily.NANODET_PLUS;
		return Nms.suppress(boxes, detectConfig.getNmsThreshold(), classAgnostic,
			detectConfig.getMaxDetections());
	}

	private static float clamp(float value, float min, float max) {
		return value < min ? min : (value > max ? max : value);
	}

}
