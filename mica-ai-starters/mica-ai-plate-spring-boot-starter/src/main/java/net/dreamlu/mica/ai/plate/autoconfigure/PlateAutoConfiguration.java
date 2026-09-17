/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.plate.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.plate.config.PlateConfig;
import net.dreamlu.mica.ai.plate.pipeline.PlatePipeline;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;

/**
 * mica-ai-plate Spring Boot 自动装配（基于 mica-auto）。
 */
@Slf4j
@Component
@EnableConfigurationProperties(PlateProperties.class)
@ConditionalOnClass(PlatePipeline.class)
@ConditionalOnProperty(prefix = "mica.ai.plate", name = "enabled", havingValue = "true", matchIfMissing = true)
public class PlateAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean
	public PlatePipeline platePipeline(PlateProperties properties) {
		PlateConfig config = PlateConfig.builder()
			.modelVersion(properties.getModelVersion())
			.detectionModelPath(properties.getDetectionModelPath())
			.recognitionModelPath(properties.getRecognitionModelPath())
			.classificationModelPath(properties.getClassificationModelPath())
			.detectionInputSize(properties.getDetectionInputSize())
			.recognitionInputHeight(properties.getRecognitionInputHeight())
			.recognitionInputWidth(properties.getRecognitionInputWidth())
			.classificationInputSize(properties.getClassificationInputSize())
			.detectionConfidenceThreshold(properties.getDetectionConfidenceThreshold())
			.detectionNmsThreshold(properties.getDetectionNmsThreshold())
			.maxPlates(properties.getMaxPlates())
			.onnx(properties.getOnnx())
			.build();
		return PlatePipeline.create(config);
	}
}
