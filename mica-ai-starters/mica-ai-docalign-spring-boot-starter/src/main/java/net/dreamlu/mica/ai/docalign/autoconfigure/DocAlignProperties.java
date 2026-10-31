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
package net.dreamlu.mica.ai.docalign.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-docalign 配置属性，对应 {@code mica.ai.docalign} 前缀。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.docalign")
public class DocAlignProperties {

	/**
	 * 是否启用自动装配（false 时不装配 DocAlignEngine Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 模型标识，仅用于日志展示，不参与路径解析。
	 */
	private String modelVersion = "lcnet100_h_e_bifpn_256_fp32";

	/**
	 * 模型 ONNX 路径（必填，支持 classpath: 前缀）。
	 *
	 * <p>模型随仓库分发但不打包进 jar，必须显式指向本地文件，
	 * 例如 {@code model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx}。
	 */
	private String modelPath;

	/**
	 * 模型输入宽度，必须与模型张量形状一致。
	 */
	private int inputWidth = 256;

	/**
	 * 模型输入高度，必须与模型张量形状一致。
	 */
	private int inputHeight = 256;

	/**
	 * 热力图二值化阈值（0~1，上游默认 0.3）。
	 */
	private float heatmapThreshold = 0.3f;

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
