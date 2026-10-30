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
package net.dreamlu.mica.ai.enhance.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-enhance 配置属性，对应 {@code mica.ai.enhance} 前缀。
 *
 * <p>切换同族模型时 {@code scale} 必须与实际模型的放大倍数一致，
 * 不一致时推理期直接抛异常。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.enhance")
public class EnhanceProperties {

	/**
	 * 是否启用自动装配（false 时不装配 EnhanceEngine Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 模型标识（对应 model-tools/enhance/models/ 下的文件名前缀）。
	 */
	private String modelVersion = "realesr-general-x4v3";

	/**
	 * 模型 ONNX 路径（支持 classpath: 前缀）；为空时按 {@code model-version} 解析。
	 */
	private String modelPath;

	/**
	 * 放大倍数；必须等于模型实际输出/输入比值。
	 */
	private int scale = 4;

	/**
	 * 单次推理的 tile 边长（原图像素）；图像小于该值时整图推理。
	 */
	private int tileSize = 256;

	/**
	 * tile 重叠边宽（原图像素），用于消除拼接接缝。
	 */
	private int tileOverlap = 16;

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
