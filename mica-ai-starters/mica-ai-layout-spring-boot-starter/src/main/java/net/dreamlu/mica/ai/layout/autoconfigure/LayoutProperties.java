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

	/**
	 * 是否启用文档版面分析（false 时不装配 LayoutPipeline Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 模型版本（对应 model-tools/layout/models/{modelVersion}/ 目录名）。
	 */
	private String modelVersion = "v3";

	/**
	 * PP-DocLayoutV3 模型 ONNX 路径（支持 classpath: 前缀；⚠️ 125MB 不入库，需本地准备）。
	 */
	private String modelPath;

	/**
	 * letterbox 输入边长上限，超过等比缩放；需与模型导出的输入边长一致，不一致启动时快速失败。
	 */
	private int maxSideLength = 800;

	/**
	 * 默认类别置信度阈值（0~1）。
	 */
	private float scoreThreshold = 0.4f;

	/**
	 * 按类别索引覆盖置信度阈值（key 为模型类别索引，value 为该类阈值），未配置的类别走 scoreThreshold。
	 */
	private Map<Integer, Float> classScoreThresholds = new HashMap<>();

	/**
	 * 是否对版面区域做 NMS 去重。
	 */
	private boolean layoutNms = true;

	/**
	 * 同类别 NMS IoU 阈值。
	 */
	private float nmsThreshold = 0.6f;

	/**
	 * 跨类别 NMS IoU 阈值（官方默认 0.98，仅过滤几乎完全重叠的跨类框）。
	 */
	private float nmsDiffClassThreshold = 0.98f;

	/**
	 * 单次推理最多保留的版面区域数。
	 */
	private int maxDetections = 100;

	/**
	 * 归一化均值（RGB 3 通道，官方 PP-DocLayoutV3 导出值）。
	 */
	private float[] mean = new float[]{0.8286f, 0.8281f, 0.8282f};

	/**
	 * 归一化标准差（RGB 3 通道，官方 PP-DocLayoutV3 导出值）。
	 */
	private float[] std = new float[]{0.1889f, 0.1889f, 0.1889f};

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
