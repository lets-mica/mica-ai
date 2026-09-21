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
package net.dreamlu.mica.ai.matting.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.matting.config.MattingInterpolation;
import net.dreamlu.mica.ai.matting.config.MattingOutputSelect;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-matting 配置属性，对应 {@code mica.ai.matting} 前缀。
 *
 * <p>换模型（如 {@code u2net} / {@code u2net_human_seg}）只需改
 * {@code model-path} 指向本地模型文件，其余契约参数三者一致，无需调整。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.matting")
public class MattingProperties {

	/**
	 * 是否启用抠图（false 时不装配 MattingEngine Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 模型标识（对应 model-tools/matting/models/ 下的文件名前缀）。
	 */
	private String modelVersion = "u2netp";

	/**
	 * 模型 ONNX 路径（支持 classpath: 前缀）。
	 *
	 * <p>任意 U²-Net 族 ONNX 均可：指向本地绝对路径即可接入超出 50MB 入库上限的模型
	 * （如 u2net / u2net_human_seg，各约 168MB）。
	 */
	private String modelPath;

	/**
	 * 模型输入边长；U²-Net 族官方导出为 320，与模型不一致时启动即失败。
	 */
	private int inputSize = 320;

	/**
	 * 输出节点选择策略：AUTO（默认）/ FIRST / D0。
	 */
	private MattingOutputSelect outputSelect = MattingOutputSelect.DEFAULT;

	/**
	 * 归一化均值（RGB 3 通道，ImageNet 统计量）。
	 */
	private float[] mean = new float[]{0.485f, 0.456f, 0.406f};

	/**
	 * 归一化标准差（RGB 3 通道，ImageNet 统计量）。
	 */
	private float[] std = new float[]{0.229f, 0.224f, 0.225f};

	/**
	 * 掩码缩放插值方式：LINEAR（默认）/ NEAREST / CUBIC。
	 */
	private MattingInterpolation interpolation = MattingInterpolation.LINEAR;

	/**
	 * 二值掩码阈值（0~1），仅二值输出接口使用。
	 */
	private float binaryThreshold = 0.5f;

	/**
	 * 是否对模型输出做 min-max 拉伸到 [0,1]（rembg 参考行为）。
	 */
	private boolean minMaxNormalize = true;

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
