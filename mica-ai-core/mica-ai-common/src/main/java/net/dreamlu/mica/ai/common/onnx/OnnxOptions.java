/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.onnx;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * ONNX 会话选项（跨能力通用基础设施）。
 *
 * @author L.cm
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class OnnxOptions {

	@Builder.Default
	private int intraOpNumThreads = 0;

	@Builder.Default
	private int interOpNumThreads = 0;

	@Builder.Default
	private boolean gpu = false;

	@Builder.Default
	private int cudaDeviceId = 0;

	public static OnnxOptions defaults() {
		return OnnxOptions.builder().build();
	}
}
