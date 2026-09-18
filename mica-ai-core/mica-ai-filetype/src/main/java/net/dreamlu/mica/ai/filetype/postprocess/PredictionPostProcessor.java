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
package net.dreamlu.mica.ai.filetype.postprocess;

import net.dreamlu.mica.ai.filetype.config.PredictionMode;
import net.dreamlu.mica.ai.filetype.config.ContentTypeRegistry;
import net.dreamlu.mica.ai.filetype.config.ModelConfig;
import net.dreamlu.mica.ai.filetype.model.ContentTypeLabel;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;

/**
 * 预测后处理，对齐 magika Python 版
 * {@code _get_output_label_from_dl_label_and_score}。
 *
 * <p>先应用 overwrite_map，再按 {@link PredictionMode} 判定是否信任模型输出；
 * 不信任时按 kb 的 is_text 信息降级为 {@code txt} / {@code unknown}。
 *
 * <p>线程安全：构造后所有字段均为 final，无状态变更，可作为单例 Bean 共享。
 */
public class PredictionPostProcessor {

	private final ModelConfig modelConfig;
	private final ContentTypeRegistry registry;
	private final PredictionMode predictionMode;

	public PredictionPostProcessor(ModelConfig modelConfig,
								   ContentTypeRegistry registry,
								   PredictionMode predictionMode) {
		this.modelConfig = Objects.requireNonNull(modelConfig, "ModelConfig must not be null");
		this.registry = Objects.requireNonNull(registry, "ContentTypeRegistry must not be null");
		this.predictionMode = Objects.requireNonNull(predictionMode, "PredictionMode must not be null");
	}

	/**
	 * 解析最终对外标签。
	 *
	 * @param dlLabel 模型直接 argmax 输出的标签
	 * @param score   该标签的 softmax 概率
	 * @return 经 overwrite_map + 置信度阈值后处理的标签；阈值不通过则回退 {@code txt} / {@code unknown}
	 */
	public String resolveOutputLabel(String dlLabel, float score) {
		Map<String, String> overwriteMap = modelConfig.getOverwriteMap();
		if (overwriteMap == null) {
			overwriteMap = Collections.emptyMap();
		}
		String outputLabel = overwriteMap.getOrDefault(dlLabel, dlLabel);

		if (predictionMode == PredictionMode.BEST_GUESS) {
			return outputLabel;
		}
		if (predictionMode == PredictionMode.HIGH_CONFIDENCE
			&& score >= highConfidenceThreshold(dlLabel)) {
			return outputLabel;
		}
		if (predictionMode == PredictionMode.MEDIUM_CONFIDENCE
			&& score >= modelConfig.getMediumConfidenceThreshold()) {
			return outputLabel;
		}
		return registry.get(outputLabel).isText() ? ContentTypeLabel.TXT : ContentTypeLabel.UNKNOWN;
	}

	private float highConfidenceThreshold(String dlLabel) {
		Map<String, Float> thresholds = modelConfig.getThresholds();
		if (thresholds != null) {
			Float threshold = thresholds.get(dlLabel);
			if (threshold != null) {
				return threshold;
			}
		}
		return modelConfig.getMediumConfidenceThreshold();
	}
}
