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
package net.dreamlu.mica.ai.humanseg.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-humanseg 配置属性，对应 {@code mica.ai.humanseg} 前缀。
 *
 * <p>切换同族模型（如竖版 Portrait-PP-HumanSegV2-Lite）时 {@code input-width}
 * 与 {@code input-height} 必须成对改成模型实际的输入宽高。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.humanseg")
public class HumanSegProperties {

	/**
	 * 是否启用自动装配（false 时不装配 HumanSegEngine Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 模型标识，仅用于日志展示，不参与路径解析。
	 */
	private String modelVersion = "human_segmentation_pphumanseg_2023mar";

	/**
	 * 模型 ONNX 路径（必填，支持 classpath: 前缀）。
	 *
	 * <p>模型随仓库分发但不打包进 jar，必须显式指向本地文件，
	 * 例如 {@code model-tools/humanseg/models/human_segmentation_pphumanseg_2023mar.onnx}。
	 */
	private String modelPath;

	/**
	 * 模型输入宽度，必须与模型张量形状一致。
	 */
	private int inputWidth = 192;

	/**
	 * 模型输入高度，必须与模型张量形状一致。
	 */
	private int inputHeight = 192;

	/**
	 * 归一化均值（RGB 3 通道，PaddleSeg 默认 0.5）。
	 */
	private float[] mean = new float[]{0.5f, 0.5f, 0.5f};

	/**
	 * 归一化标准差（RGB 3 通道，PaddleSeg 默认 0.5）。
	 */
	private float[] std = new float[]{0.5f, 0.5f, 0.5f};

	/**
	 * 二值掩码阈值（0~1），仅二值输出接口使用。
	 */
	private float binaryThreshold = 0.5f;

	/**
	 * 纯色底输出时的默认底色（RGB，0~255）。
	 */
	private int[] backgroundColor = new int[]{255, 255, 255};

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
