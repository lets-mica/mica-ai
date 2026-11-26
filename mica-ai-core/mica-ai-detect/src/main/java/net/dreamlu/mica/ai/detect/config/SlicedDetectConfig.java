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
package net.dreamlu.mica.ai.detect.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.detect.model.SliceMergeStrategy;

/**
 * 切片推理配置（切片边长 / 重叠率 / 切片数上限 / 跨片合并策略）。
 *
 * <p>本配置**委托** {@link DetectConfig}，不重复声明模型路径、阈值等检测参数 ——
 * 切片推理只是「在整图检测之上加一层切片编排」。
 *
 * <p><b>切片边长不是配置项</b>：它恒等于 {@link DetectConfig#getInputSize()}
 * （模型张量边长）。模型输入尺寸固定，切片边长若更小，letterbox 会把切片放大，
 * 小目标优势归零；若更大则浪费算力且超出模型输入。
 *
 * <pre>{@code
 * SlicedDetectConfig sliced = SlicedDetectConfig.builder()
 *     .detectConfig(DetectConfig.builder()
 *         .modelPath("model-tools/detect/models/object_detection_nanodet_2022nov.onnx")
 *         .build())
 *     .overlapRatio(0.25f)
 *     .maxSlices(64)
 *     .mergeStrategy(SliceMergeStrategy.IOU_MERGE)
 *     .build();
 * }</pre>
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class SlicedDetectConfig {

	/**
	 * 委托的整图检测配置（模型路径 / 族 / 阈值 / 标签等），必填。
	 */
	private DetectConfig detectConfig;

	/**
	 * 切片重叠比例，取值 {@code (0, 1)}，默认 0.25。
	 *
	 * <p>重叠是必需的：没有重叠时，位于切片边界上的物体会被切成两半，
	 * 两半的 IoU 常常低于 NMS 阈值，跨片合并时不会被抑制，输出「半个物体」。
	 */
	@Builder.Default
	private float overlapRatio = 0.25f;

	/**
	 * 单帧切片数上限，默认 64。
	 *
	 * <p>切片推理的算力开销约为「整图 × 切片数」，大图容易把算力打爆。
	 * 超过上限时直接抛异常，并在消息里给出可操作的调整建议。
	 */
	@Builder.Default
	private int maxSlices = 64;

	/**
	 * 跨片合并策略，默认 {@link SliceMergeStrategy#IOU_MERGE}。
	 */
	@Builder.Default
	private SliceMergeStrategy mergeStrategy = SliceMergeStrategy.IOU_MERGE;

	/**
	 * 切片边长（恒等于模型输入边长）。
	 *
	 * @return 切片边长（像素）
	 */
	public int resolveSliceSize() {
		return detectConfig.getInputSize();
	}

	/**
	 * 切片步进（重叠像素 = 边长 - 步进）。
	 *
	 * @return 步进（像素，至少为 1）
	 */
	public int resolveStep() {
		int size = resolveSliceSize();
		int overlap = Math.round(size * overlapRatio);
		int step = size - overlap;
		return Math.max(1, step);
	}

	/**
	 * 校验配置合法性，非法时抛出 {@link MicaAiException}。
	 *
	 * @throws MicaAiException 任一配置项非法时抛出，错误码为 {@link ErrorCode#ILLEGAL_ARGUMENT}
	 */
	public void validate() {
		if (detectConfig == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "detectConfig 不能为空");
		}
		detectConfig.validate();
		if (overlapRatio <= 0f || overlapRatio >= 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"overlapRatio 必须在 (0, 1) 之间，当前为 " + overlapRatio);
		}
		if (maxSlices <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "maxSlices 必须为正数");
		}
		if (mergeStrategy == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "mergeStrategy 不能为空");
		}
	}

}
