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

	/**
	 * 是否启用车牌识别（false 时不装配 PlatePipeline Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 模型版本（对应 model-tools/plate/models/ 的版本标识，如 20230229）。
	 */
	private String modelVersion = "20230229";

	/**
	 * 车牌检测模型 ONNX 路径（支持 classpath: 前缀，未配置时启动失败）。
	 */
	private String detectionModelPath;

	/**
	 * 车牌字符识别（CRNN）模型 ONNX 路径（支持 classpath: 前缀，未配置时启动失败）。
	 */
	private String recognitionModelPath;

	/**
	 * 车牌颜色分类模型 ONNX 路径（支持 classpath: 前缀，未配置时启动失败）。
	 */
	private String classificationModelPath;

	/**
	 * 检测模型输入边长（320 或 640，需与模型实际输入一致）。
	 */
	private int detectionInputSize = 320;

	/**
	 * 识别模型输入高度（HyperLPR3 默认 48）。
	 */
	private int recognitionInputHeight = 48;

	/**
	 * 识别模型输入宽度（HyperLPR3 默认 160）。
	 */
	private int recognitionInputWidth = 160;

	/**
	 * 颜色分类模型输入边长（HyperLPR3 默认 96）。
	 */
	private int classificationInputSize = 96;

	/**
	 * 检测置信度阈值（0~1，低于该值的候选框丢弃）。
	 */
	private float detectionConfidenceThreshold = 0.25f;

	/**
	 * 检测 NMS IoU 阈值。
	 */
	private float detectionNmsThreshold = 0.5f;

	/**
	 * 单张图片最多识别的车牌数。
	 */
	private int maxPlates = 5;

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
