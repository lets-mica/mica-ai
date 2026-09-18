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
package net.dreamlu.mica.ai.plate.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-plate 配置属性，对应 {@code mica.ai.plate} 前缀。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.plate")
public class PlateProperties {

	private boolean enabled = true;
	private String modelVersion = "20230229";
	private String detectionModelPath;
	private String recognitionModelPath;
	private String classificationModelPath;
	private int detectionInputSize = 320;
	private int recognitionInputHeight = 48;
	private int recognitionInputWidth = 160;
	private int classificationInputSize = 96;
	private float detectionConfidenceThreshold = 0.25f;
	private float detectionNmsThreshold = 0.5f;
	private int maxPlates = 5;
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
