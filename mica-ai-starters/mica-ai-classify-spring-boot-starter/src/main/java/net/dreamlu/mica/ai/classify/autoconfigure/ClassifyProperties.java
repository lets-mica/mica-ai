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
package net.dreamlu.mica.ai.classify.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.classify.config.ClassifyChannelOrder;
import net.dreamlu.mica.ai.classify.config.ClassifyInterpolation;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-classify 配置属性，对应 {@code mica.ai.classify} 前缀。
 *
 * <p>一个引擎实例绑定「一个模型 + 一份 label_list」，两者必须成对配置；
 * 需要同时使用多个 PULC 模型时，分别声明多个 {@code ClassifyEngine} Bean。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.classify")
public class ClassifyProperties {

	/**
	 * 是否启用自动装配（false 时不装配 ClassifyEngine Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 模型标识（对应 model-tools/classify/models/ 下的文件名前缀）。
	 */
	private String modelVersion = "text_image_orientation";

	/**
	 * 模型 ONNX 路径（支持 classpath: 前缀）；为空时按 {@code model-version} 解析。
	 */
	private String modelPath;

	/**
	 * 与模型强耦合的 label_list 文件路径（必填，无 classpath 兜底）。
	 */
	private String labelsPath;

	/**
	 * 模型输入宽度（NCHW 最后一维），官方 PULC 导出为 224。
	 */
	private int inputWidth = 224;

	/**
	 * 模型输入高度（NCHW 倒数第二维），官方 PULC 导出为 224。
	 */
	private int inputHeight = 224;

	/**
	 * 短边缩放目标；官方 PULC 默认 256，{@code table_attribute} 这类直接 resize 的模型置 0。
	 */
	private int resizeShort = 256;

	/**
	 * 输入通道顺序：RGB（默认，PaddleClas 约定）/ BGR。
	 */
	private ClassifyChannelOrder channelOrder = ClassifyChannelOrder.RGB;

	/**
	 * 归一化均值（RGB 3 通道，PaddleClas NormalizeImage 默认值）。
	 */
	private float[] mean = new float[]{0.485f, 0.456f, 0.406f};

	/**
	 * 归一化标准差（RGB 3 通道，PaddleClas NormalizeImage 默认值）。
	 */
	private float[] std = new float[]{0.229f, 0.224f, 0.225f};

	/**
	 * 缩放插值方式：LINEAR（默认）/ NEAREST / CUBIC。
	 */
	private ClassifyInterpolation interpolation = ClassifyInterpolation.LINEAR;

	/**
	 * 是否多标签（每个输出独立二分类，如 {@code table_attribute}）。
	 */
	private boolean multiLabel = false;

	/**
	 * 模型输出是否已是概率；PULC 官方导出已内置 softmax / sigmoid，默认 true。
	 */
	private boolean outputIsProbability = true;

	/**
	 * 返回前 K 个类别（单标签模式有意义，多标签模式最多返回全部类别）。
	 */
	private int topK = 1;

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
