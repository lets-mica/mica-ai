/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.onnx;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

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
	private Device device = Device.CPU;

	@Builder.Default
	private int cudaDeviceId = 0;

	@Builder.Default
	private int intraOpNumThreads = 0;

	@Builder.Default
	private int interOpNumThreads = 0;

	@Builder.Default
	private GraphOptimizationLevel graphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

	@Builder.Default
	private ExecutionMode executionMode = ExecutionMode.ORT_PARALLEL;

	public enum Device {
		CPU,
		GPU
	}

	public enum GraphOptimizationLevel {
		ORT_DISABLE_ALL,
		ORT_ENABLE_BASIC,
		ORT_ENABLE_EXTENDED,
		ORT_ENABLE_ALL
	}

	public enum ExecutionMode {
		ORT_SEQUENTIAL,
		ORT_PARALLEL
	}

	public static OrtSessionOptions defaults() {
		return OrtSessionOptions.builder().build();
	}
}