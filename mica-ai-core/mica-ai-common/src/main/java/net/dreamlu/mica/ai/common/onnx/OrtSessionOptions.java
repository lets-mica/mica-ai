/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.onnx;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Collections;
import java.util.Map;

/**
 * ONNX Runtime 会话基础配置（跨能力共享）。
 *
 * <p>统一描述所有模型共享的 {@link ai.onnxruntime.OrtSession.SessionOptions} 行为，
 * 由 Spring Boot Starter 从 {@code application.yml} 的 {@code mica.ai.<cap>.onnx} 映射，
 * 或直接通过 {@link #builder()} 构造后塞进各能力的模型配置。
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OrtSessionOptions {

	@Builder.Default
	private OrtDevice device = OrtDevice.CPU;

	@Builder.Default
	private int cudaDeviceId = 0;

	@Builder.Default
	private int intraOpNumThreads = 0;

	@Builder.Default
	private int interOpNumThreads = 0;

	@Builder.Default
	private OrtGraphOptimizationLevel graphOptimizationLevel = OrtGraphOptimizationLevel.ENABLE_ALL;

	@Builder.Default
	private OrtExecutionMode executionMode = OrtExecutionMode.PARALLEL;

	@Builder.Default
	private boolean enableCpuMemArena = false;

	@Builder.Default
	private boolean enableMemoryPattern = false;

	@Builder.Default
	private Map<String, Long> symbolicDimensionValues = Collections.emptyMap();

	public static OrtSessionOptions defaults() {
		return OrtSessionOptions.builder().build();
	}
}
