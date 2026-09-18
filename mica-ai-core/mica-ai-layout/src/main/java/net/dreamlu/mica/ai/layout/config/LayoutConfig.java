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
import java.util.Set;

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
	 * 相对阈值系数（相对 top1 分数的下限），0 表示不启用。
	 *
	 * <p>启用后有效阈值为 {@code max(scoreThreshold, top1Score * scoreRatio)}，
	 * 用于抑制「同一页内分数悬崖下方的长尾误检」。实测（官方 demo 1654×2339）：
	 * 真实内容区分数 0.69–0.94，而 0.4370 / 0.5182 是跨页长条误检；
	 * 同时单列裁剪图整体分数仅 ~0.31–0.37，纯绝对阈值无法同时兼顾两者。
	 * 取 {@code 0.6} 时四种缩放 / 裁剪变体均能完整保留真实内容并丢弃长尾。
	 */
	@Builder.Default
	private float scoreRatio = 0f;

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

	/**
	 * 不参与阅读顺序编号的标签名单，默认对齐 PaddleX
	 * {@code LayoutAnalysisProcess.SKIP_ORDER_LABELS}（11 类）。
	 *
	 * <p>语义同 PaddleX：名单内的区域 {@code readingOrder} 固定为
	 * {@link net.dreamlu.mica.ai.layout.model.LayoutResult#NO_READING_ORDER}，
	 * 且**不占用编号**；其余区域从 1 开始连续编号。
	 * 传空集合表示「所有类别都参与编号」。
	 */
	@Builder.Default
	private Set<LayoutLabel> skipOrderLabels = LayoutLabel.defaultSkipOrderLabels();

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
		if (scoreRatio < 0f || scoreRatio > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"scoreRatio 必须在 [0,1]");
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
	 * 判断某标签是否不参与阅读顺序编号。
	 *
	 * @param label 版面标签；为 {@code null} 时参与编号
	 * @return 在 {@link #skipOrderLabels} 名单内返回 {@code true}
	 */
	public boolean isSkipOrderLabel(LayoutLabel label) {
		return label != null && skipOrderLabels != null && skipOrderLabels.contains(label);
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