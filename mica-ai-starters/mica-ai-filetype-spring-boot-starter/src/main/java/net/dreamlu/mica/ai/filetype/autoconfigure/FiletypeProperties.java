/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

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
	private String predictionMode = "HIGH_CONFIDENCE";
	private String device = "cpu";
	private Onnx onnx = new Onnx();

	@Getter
	@Setter
	public static class Onnx {
		private int intraOpNumThreads = 0;
		private int interOpNumThreads = 0;
		private int cudaDeviceId = 0;
	}
}
