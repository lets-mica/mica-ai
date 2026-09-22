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
package net.dreamlu.mica.ai.textline.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.textline.config.TextLineChannelOrder;
import net.dreamlu.mica.ai.textline.config.TextLineInterpolation;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-textline 配置属性，对应 {@code mica.ai.textline} 前缀。
 *
 * <p>换模型（如轻量版 {@code PP-LCNet_x0_25_textline_ori}，约 0.96MB）只需改
 * {@code model-path}，其余契约参数一致，无需调整。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.textline")
public class TextLineProperties {

	/**
	 * 是否启用文本行方向分类（false 时不装配 TextLineEngine Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 模型标识（对应 model-tools/textline/models/ 下的文件名前缀）。
	 */
	private String modelVersion = "PP-LCNet_x1_0_textline_ori";

	/**
	 * 模型 ONNX 路径（支持 classpath: 前缀）。
	 *
	 * <p>同任务模型可直接替换，如 PP-LCNet_x0_25_textline_ori（约 0.96MB，更轻）。
	 */
	private String modelPath;

	/**
	 * 模型输入宽度（NCHW 最后一维），官方导出为 160。
	 */
	private int inputWidth = 160;

	/**
	 * 模型输入高度（NCHW 倒数第二维），官方导出为 80。
	 */
	private int inputHeight = 80;

	/**
	 * 输入通道顺序：BGR（默认，官方 PaddleX 约定）/ RGB。
	 */
	private TextLineChannelOrder channelOrder = TextLineChannelOrder.BGR;

	/**
	 * 归一化均值（3 通道，PaddleX NormalizeImage 默认值）。
	 */
	private float[] mean = new float[]{0.485f, 0.456f, 0.406f};

	/**
	 * 归一化标准差（3 通道，PaddleX NormalizeImage 默认值）。
	 */
	private float[] std = new float[]{0.229f, 0.224f, 0.225f};

	/**
	 * 缩放插值方式：LINEAR（默认）/ NEAREST / CUBIC。
	 */
	private TextLineInterpolation interpolation = TextLineInterpolation.LINEAR;

	/**
	 * 判定为「倒置」的最小概率；调高可让拿不准的行保持不动。
	 */
	private float upsideDownThreshold = 0.5f;

	/**
	 * 模型输出是否已是概率（softmax 后）；官方 ONNX 输出为原始 logits，故默认 false。
	 */
	private boolean outputIsProbability = false;

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
