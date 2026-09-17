/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.layout.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

import java.util.HashMap;
import java.util.Map;

/**
 * mica-ai-layout 配置属性，对应 {@code mica.ai.layout} 前缀。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.layout")
public class LayoutProperties {

	private boolean enabled = true;
	private String modelVersion = "v3";
	private String modelPath;
	private int maxSideLength = 800;
	private float scoreThreshold = 0.4f;
	private Map<Integer, Float> classScoreThresholds = new HashMap<>();
	private boolean layoutNms = true;
	private float nmsThreshold = 0.6f;
	private float nmsDiffClassThreshold = 0.98f;
	private int maxDetections = 100;
	private float[] mean = new float[]{0.8286f, 0.8281f, 0.8282f};
	private float[] std = new float[]{0.1889f, 0.1889f, 0.1889f};
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}