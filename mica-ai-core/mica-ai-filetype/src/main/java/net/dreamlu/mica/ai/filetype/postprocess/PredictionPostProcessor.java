/*
 * Copyright (c) 2024-2026 mica-ai
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
