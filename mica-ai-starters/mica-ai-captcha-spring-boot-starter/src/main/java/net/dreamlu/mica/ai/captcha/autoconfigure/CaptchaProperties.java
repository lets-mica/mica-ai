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
package net.dreamlu.mica.ai.captcha.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.captcha.config.CaptchaInterpolation;
import net.dreamlu.mica.ai.captcha.config.CaptchaNormalization;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-captcha 配置属性，对应 {@code mica.ai.captcha} 前缀。
 *
 * <p><b>模型外置</b>：{@link #modelPath} 与 {@link #charsetPath} 必填，
 * 两者都不随 jar 分发，需指向本地文件（如
 * {@code model-tools/captcha/models/ddddocr.onnx} 与同目录的 {@code charset.txt}）。
 * 未配置时启动即失败并给出明确提示，而不是等到第一次识别才报错。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.captcha")
public class CaptchaProperties {

	/**
	 * 是否启用验证码识别（false 时不装配 CaptchaEngine Bean）。
	 */
	private boolean enabled = true;

	/**
	 * ONNX 模型路径（必填，支持 classpath: 前缀，其余按文件系统路径处理）。
	 *
	 * <p>模型随仓库分发但不打包进 jar，必须显式指向本地文件，
	 * 例如 {@code model-tools/captcha/models/ddddocr.onnx}。
	 */
	private String modelPath;

	/**
	 * 字符集文件路径（UTF-8 文本，每个字符占一个索引；支持 classpath: 前缀）。
	 */
	private String charsetPath;

	/**
	 * 模型输入高度（NCHW 倒数第二维），官方 ddddocr 模型为 64，宽度按原图比例自适应。
	 */
	private int inputHeight = 64;

	/**
	 * 输入归一化方式：DIV_255（默认）/ MEAN_0_5（对齐 ddddocr Python 原版）。
	 */
	private CaptchaNormalization normalization = CaptchaNormalization.DIV_255;

	/**
	 * 缩放插值方式：BILINEAR（默认）/ NEAREST / BICUBIC。
	 */
	private CaptchaInterpolation interpolation = CaptchaInterpolation.BILINEAR;

	/**
	 * 是否把带透明通道的 PNG 的透明像素按黑色合成。
	 */
	private boolean pngFix = true;

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

}
