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
package net.dreamlu.mica.ai.captcha.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

import java.nio.file.Files;
import java.nio.file.Paths;

/**
 * 通用验证码识别配置（模型路径、字符集路径、输入高度、归一化、插值等）。
 *
 * <p><b>模型外置</b>：本模块<b>不把模型打进 jar</b>，{@link #modelPath} 与
 * {@link #charsetPath} 均为必填项，由使用者指向本地文件（仓库内自带的示例路径为
 * {@code model-tools/captcha/models/ddddocr.onnx} 与同目录下的 {@code charset.txt}）。
 * 字符集与模型必须成对使用：字符集第 <i>i</i> 个字符对应模型输出的第 <i>i</i> 类，
 * 索引 0 固定为 CTC 的 blank。
 *
 * <p><b>可插拔</b>：契约参数（输入高度、归一化方式、插值方式）全部显式化，
 * 换用同族的其它 CRNN 验证码模型时，只需改 {@link #modelPath} + {@link #charsetPath}，
 * 必要时调整 {@link #inputHeight} / {@link #normalization}，<b>Java 代码零改动</b>。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class CaptchaConfig {

	/**
	 * ONNX 模型路径；支持 {@code classpath:} 前缀，其余按文件系统绝对 / 相对路径处理
	 */
	private String modelPath;

	/**
	 * 字符集文件路径（UTF-8 文本，每个字符占一个索引，忽略换行符）；
	 * 支持 {@code classpath:} 前缀
	 */
	private String charsetPath;

	/**
	 * 模型输入高度（NCHW 倒数第二维），官方 {@code ddddocr} 模型为 64，宽度按原图比例自适应
	 */
	@Builder.Default
	private int inputHeight = 64;

	/**
	 * 输入归一化方式；默认 {@link CaptchaNormalization#DIV_255}
	 */
	@Builder.Default
	private CaptchaNormalization normalization = CaptchaNormalization.DIV_255;

	/**
	 * 缩放插值方式；默认 {@link CaptchaInterpolation#BILINEAR}
	 */
	@Builder.Default
	private CaptchaInterpolation interpolation = CaptchaInterpolation.BILINEAR;

	/**
	 * 是否对带透明通道的 PNG 做预处理（透明像素按黑色处理）。
	 *
	 * <p>与 PIL {@code Image.convert('L')} 的行为一致：RGBA 转灰度时忽略 alpha 通道，
	 * 而 PNG 中全透明像素的 RGB 通常为 0，故表现为「透明处为黑」。
	 */
	@Builder.Default
	private boolean pngFix = true;

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）
	 */
	@Builder.Default
	private OrtSessionOptions onnx = OrtSessionOptions.defaults();

	/**
	 * 校验配置合法性，非法时抛出 {@link MicaAiException}。
	 *
	 * @throws MicaAiException 任一配置项非法时抛出，错误码为 {@link ErrorCode#ILLEGAL_ARGUMENT}；
	 *                         模型 / 字符集文件不存在时为 {@link ErrorCode#NOT_FOUND}
	 */
	public void validate() {
		if (modelPath == null || modelPath.trim().isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"modelPath 不能为空：验证码模型为外置配置，不随 jar 分发，"
					+ "请指向本地模型文件（如 model-tools/captcha/models/ddddocr.onnx）");
		}
		if (charsetPath == null || charsetPath.trim().isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"charsetPath 不能为空：字符集与模型必须成对使用，"
					+ "请指向本地字符集文件（如 model-tools/captcha/models/charset.txt）");
		}
		if (inputHeight <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "inputHeight 必须为正数");
		}
		if (normalization == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "normalization must not be null");
		}
		if (interpolation == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "interpolation must not be null");
		}
		assertFileExists(modelPath, "模型");
		assertFileExists(charsetPath, "字符集");
	}

	/**
	 * 解析实际使用的模型路径。
	 *
	 * @return 配置的模型路径（本模块模型外置，不提供内置 classpath 兜底）
	 */
	public String resolveModelPath() {
		return modelPath;
	}

	private static void assertFileExists(String path, String label) {
		if (OnnxModelSession.isClasspath(path)) {
			return;
		}
		if (!Files.exists(Paths.get(path))) {
			throw new MicaAiException(ErrorCode.NOT_FOUND,
				label + "文件不存在: " + path
					+ "（验证码模型为外置配置，需自行放置模型与字符集文件）");
		}
	}
}
