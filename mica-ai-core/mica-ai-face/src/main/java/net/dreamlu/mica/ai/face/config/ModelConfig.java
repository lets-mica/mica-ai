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
package net.dreamlu.mica.ai.face.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * 模型加载配置（immutable）。
 */
@Getter
@Builder(toBuilder = true)
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
