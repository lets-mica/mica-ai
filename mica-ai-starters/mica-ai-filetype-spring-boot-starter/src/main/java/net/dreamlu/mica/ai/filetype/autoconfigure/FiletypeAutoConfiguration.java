/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.onnx.OnnxOptions;
import net.dreamlu.mica.ai.filetype.FiletypeConfig;
import net.dreamlu.mica.ai.filetype.FiletypeDetector;
import net.dreamlu.mica.ai.filetype.PredictionMode;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;

/**
 * mica-ai-filetype Spring Boot 自动装配（基于 mica-auto）。
 *
 * <p>由 {@code mica-auto} 扫描本类上的 {@link Component} 注解，自动生成
 * {@code META-INF/spring.factories} 中的 {@code EnableAutoConfiguration} 条目。
 */
@Slf4j
@Component
@EnableConfigurationProperties(FiletypeProperties.class)
@ConditionalOnClass(FiletypeDetector.class)
@ConditionalOnProperty(prefix = "mica.ai.filetype", name = "enabled", havingValue = "true", matchIfMissing = true)
public class FiletypeAutoConfiguration {

	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public FiletypeDetector filetypeDetector(FiletypeProperties properties) {
		FiletypeConfig config = FiletypeConfig.builder()
			.modelVersion(properties.getModelVersion())
			.modelPath(properties.getModelPath())
			.configPath(properties.getConfigPath())
			.contentTypesPath(properties.getContentTypesPath())
			.predictionMode(parseMode(properties.getPredictionMode()))
			.onnx(OnnxOptions.builder()
				.intraOpNumThreads(properties.getOnnx().getIntraOpNumThreads())
				.interOpNumThreads(properties.getOnnx().getInterOpNumThreads())
				.gpu("gpu".equalsIgnoreCase(properties.getDevice()))
				.cudaDeviceId(properties.getOnnx().getCudaDeviceId())
				.build())
			.build();
		return FiletypeDetector.create(config);
	}

	private static PredictionMode parseMode(String s) {
		if (s == null || s.trim().isEmpty()) {
			return PredictionMode.HIGH_CONFIDENCE;
		}
		return PredictionMode.valueOf(s.trim().toUpperCase());
	}
}
