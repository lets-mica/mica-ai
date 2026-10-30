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
package net.dreamlu.mica.ai.detect.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.detect.config.DetectModelFamily;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-detect 配置属性，对应 {@code mica.ai.detect} 前缀。
 *
 * <p>切换模型族时 {@code family} 与 {@code input-size} 必须成对修改：
 * NanoDet-Plus 为 {@code 416}，YOLOX-s 为 {@code 640}。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.detect")
public class DetectProperties {

	/**
	 * 是否启用目标检测（false 时不装配 DetectEngine Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 模型 ONNX 路径（支持 classpath: 前缀）；为空时按 {@code family} 解析内置 classpath 路径。
	 */
	private String modelPath;

	/**
	 * 模型族：NANODET_PLUS（默认）/ YOLOX_S。
	 */
	private DetectModelFamily family = DetectModelFamily.NANODET_PLUS;

	/**
	 * 模型输入边长，必须与模型张量形状一致（NanoDet 416 / YOLOX 640）。
	 */
	private int inputSize = 416;

	/**
	 * 置信度阈值，低于该值的候选框丢弃。
	 */
	private float scoreThreshold = 0.35f;

	/**
	 * NMS 的 IoU 阈值。
	 */
	private float nmsThreshold = 0.6f;

	/**
	 * 单帧最多输出的检测框数量。
	 */
	private int maxDetections = 100;

	/**
	 * 类别名（按 classId 索引）；为空时使用内置 COCO 80 类。
	 */
	private String[] labels;

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
