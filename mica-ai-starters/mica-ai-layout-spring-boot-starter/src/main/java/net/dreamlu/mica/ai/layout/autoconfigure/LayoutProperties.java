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