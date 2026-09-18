/*
 * Copyright (c) 2019-2029, Dreamlu 卢春梦 (596392912@qq.com & dreamlu.net).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
 * <p>统一描述所有模型共享的 ONNX Runtime {@code OrtSession.SessionOptions} 行为，
 * 由 Spring Boot Starter 从 {@code application.yml} 的 {@code mica.ai.<cap>.onnx} 映射，
 * 或直接通过 builder 构造后塞进各能力的模型配置。
 *
 * <p>不可变：字段均为 final；业务代码如需调整请使用 {@code .toBuilder()} 派生副本。
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
