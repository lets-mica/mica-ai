/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import net.dreamlu.mica.ai.face.onnx.OrtSessionOptions;

/**
 * 模型加载配置。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ModelConfig {

	private String detectionModelPath;
	private String recognitionModelPath;
	private String livenessModelPath;

	@Builder.Default
	private float detectionThreshold = 0.9f;

	@Builder.Default
	private float nmsThreshold = 0.3f;

	private OrtSessionOptions onnx;

	@Builder.Default
	private float verifyThreshold = 0.35f;
}