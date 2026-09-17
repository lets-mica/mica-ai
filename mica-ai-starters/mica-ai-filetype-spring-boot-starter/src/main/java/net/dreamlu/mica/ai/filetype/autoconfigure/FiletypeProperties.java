/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.filetype.config.PredictionMode;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-filetype 配置属性，对应 {@code mica.ai.filetype} 前缀。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.filetype")
public class FiletypeProperties {

	private boolean enabled = true;
	private String modelVersion = "standard_v3_3";
	private String modelPath;
	private String configPath;
	private String contentTypesPath;
	private PredictionMode predictionMode = PredictionMode.HIGH_CONFIDENCE;
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
