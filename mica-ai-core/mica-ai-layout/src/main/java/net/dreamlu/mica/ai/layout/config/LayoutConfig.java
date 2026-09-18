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
package net.dreamlu.mica.ai.layout.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.layout.model.LayoutLabel;

import java.util.HashMap;
import java.util.Map;

/**
 * PP-DocLayoutV2 / V3 版面分析配置（模型路径、letterbox 尺寸、阈值、NMS 等）。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class LayoutConfig {

	private static final String MODEL_RESOURCE = "mica-ai/models/layout/%s/model.onnx";

	/**
	 * 自定义模型路径；为空时按 {@code modelVersion} 解析内置 classpath 路径
	 */
	private String modelPath;

	/**
	 * 内置模型版本（v2 / v3），默认 v3
	 */
	@Builder.Default
	private String modelVersion = "v3";

	/**
	 * letterbox 方形边长（与官方 PP-DocLayout 默认一致：800x800）
	 */
	@Builder.Default
	private int maxSideLength = 800;

	/**
	 * 全局置信度阈值，per-classMap 未配置某类时回退到此值
	 */
	@Builder.Default
	private float scoreThreshold = 0.4f;

	/**
	 * per-class 阈值；null 表示全部走 scoreThreshold
	 */
	@Builder.Default
	private Map<Integer, Float> classScoreThresholds = new HashMap<>();

	/**
	 * 是否启用 NMS（PP-DocLayoutV3 导出的 ONNX 未内置 NMS，输出恒为 300 个 query）
	 */
	@Builder.Default
	private boolean layoutNms = true;

	/**
	 * NMS 同类 IoU 阈值，对齐 PaddleX {@code nms(..., iou_same=0.6, iou_diff=0.98)}
	 */
	@Builder.Default
	private float nmsThreshold = 0.6f;

	/**
	 * NMS 异类 IoU 阈值；0.98 意味着不同类别之间几乎不互斥（版面区域允许嵌套）
	 */
	@Builder.Default
	private float nmsDiffClassThreshold = 0.98f;

	@Builder.Default
	private int maxDetections = 100;

	@Builder.Default
	private float[] mean = new float[]{0.8286f, 0.8281f, 0.8282f};

	@Builder.Default
	private float[] std = new float[]{0.1889f, 0.1889f, 0.1889f};

	@Builder.Default
	private OrtSessionOptions onnx = OrtSessionOptions.defaults();

	/**
	 * 校验配置合法性，非法时抛出 {@link MicaAiException}。
	 *
	 * @throws MicaAiException 任一配置项非法时抛出，错误码为 {@link ErrorCode#ILLEGAL_ARGUMENT}
	 */
	public void validate() {
		if (modelVersion == null || modelVersion.isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"modelVersion must not be null or empty");
		}
		if (maxSideLength <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"maxSideLength 必须为正数");
		}
		if (scoreThreshold < 0f || scoreThreshold > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"scoreThreshold 必须在 [0,1]");
		}
		if (maxDetections <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"maxDetections 必须为正数");
		}
		if (nmsThreshold < 0f || nmsThreshold > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"nmsThreshold 必须在 [0,1]");
		}
		if (nmsDiffClassThreshold < 0f || nmsDiffClassThreshold > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"nmsDiffClassThreshold 必须在 [0,1]");
		}
		if (mean == null || mean.length != 3 || std == null || std.length != 3) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"mean / std 必须为长度 3 的数组");
		}
		if (classScoreThresholds != null) {
			for (Map.Entry<Integer, Float> e : classScoreThresholds.entrySet()) {
				if (e.getKey() == null || e.getKey() < 0 || e.getKey() >= LayoutLabel.numClasses()) {
					throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
						"classScoreThresholds key 必须在 [0," + LayoutLabel.numClasses() + ")");
				}
				Float v = e.getValue();
				if (v == null || v < 0f || v > 1f) {
					throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
						"classScoreThresholds value 必须在 [0,1]");
				}
			}
		}
	}

	/**
	 * 获取指定类别的置信度阈值。
	 *
	 * @param classIndex 类别索引
	 * @return per-class 阈值；未配置时回退到全局 {@code scoreThreshold}
	 */
	public float thresholdFor(int classIndex) {
		Float v = classScoreThresholds == null ? null : classScoreThresholds.get(classIndex);
		return v == null ? scoreThreshold : v;
	}

	/**
	 * 解析实际使用的模型路径。
	 *
	 * @return 配置了 {@code modelPath} 时直接返回；否则返回内置 classpath 资源路径
	 */
	public String resolveModelPath() {
		if (modelPath != null && !modelPath.isEmpty()) {
			return modelPath;
		}
		return OnnxModelSession.CLASSPATH_PREFIX
			+ String.format(MODEL_RESOURCE, modelVersion);
	}
}