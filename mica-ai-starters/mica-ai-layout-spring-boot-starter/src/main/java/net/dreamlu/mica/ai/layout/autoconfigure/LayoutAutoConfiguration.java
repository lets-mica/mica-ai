/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.layout.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.layout.config.LayoutConfig;
import net.dreamlu.mica.ai.layout.pipeline.LayoutPipeline;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;

/**
 * mica-ai-layout Spring Boot 自动装配（基于 mica-auto）。
 */
@Slf4j
@Component
@EnableConfigurationProperties(LayoutProperties.class)
@ConditionalOnClass(LayoutPipeline.class)
@ConditionalOnProperty(prefix = "mica.ai.layout", name = "enabled", havingValue = "true", matchIfMissing = true)
public class LayoutAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean
	public LayoutPipeline layoutPipeline(LayoutProperties properties) {
		LayoutConfig config = LayoutConfig.builder()
			.modelVersion(properties.getModelVersion())
			.modelPath(properties.getModelPath())
			.maxSideLength(properties.getMaxSideLength())
			.scoreThreshold(properties.getScoreThreshold())
			.classScoreThresholds(properties.getClassScoreThresholds())
			.layoutNms(properties.isLayoutNms())
			.nmsThreshold(properties.getNmsThreshold())
			.nmsDiffClassThreshold(properties.getNmsDiffClassThreshold())
			.maxDetections(properties.getMaxDetections())
			.mean(properties.getMean())
			.std(properties.getStd())
			.onnx(properties.getOnnx())
			.build();
		return LayoutPipeline.create(config);
	}
}