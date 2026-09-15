/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.onnx;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * mica-ai-face ONNX 会话基础配置（与 mica-ai-common 解耦，避免 starter 重复字段）。
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