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
package net.dreamlu.mica.ai.filetype.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.filetype.config.PredictionMode;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

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
	private PredictionMode predictionMode = PredictionMode.HIGH_CONFIDENCE;
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
