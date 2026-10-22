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
package net.dreamlu.mica.ai.captcha;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.captcha.config.CaptchaConfig;
import net.dreamlu.mica.ai.captcha.detection.CaptchaDetector;
import net.dreamlu.mica.ai.captcha.model.CaptchaResult;
import net.dreamlu.mica.ai.captcha.util.CaptchaImageUtils;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;

import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * 通用验证码识别门面。
 *
 * <p>对外提供三种入参形态（字节 / base64 / 文件路径）与两种粒度（只要文本 / 带置信度的完整结果）：
 * <ul>
 *   <li>{@link #recognize(byte[])} / {@link #recognizeBase64(String)} / {@link #recognizePath(String)} —
 *       只返回识别文本</li>
 *   <li>{@link #recognizeDetailed(byte[])} 等 — 返回 {@link CaptchaResult}，
 *       含逐字符置信度，便于「置信度太低就判失败」的二次校验</li>
 * </ul>
 *
 * <p><b>模型外置</b>：模型与字符集文件都不打进 jar，须通过 {@link CaptchaConfig} 指定本地路径：
 *
 * <pre>{@code
 * CaptchaConfig config = CaptchaConfig.builder()
 *     .modelPath("model-tools/captcha/models/ddddocr.onnx")
 *     .charsetPath("model-tools/captcha/models/charset.txt")
 *     .build();
 * try (CaptchaEngine engine = CaptchaEngine.create(config)) {
 *     String text = engine.recognize(captchaBytes);
 * }
 * }</pre>
 *
 * <p>线程安全：内部 detector + ONNX session 线程安全；本类自身仅含构造时冻结的不可变字段，
 * 可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class CaptchaEngine implements AutoCloseable {

	private final CaptchaDetector detector;
	/**
	 * -- GETTER --
	 *  获取当前配置。
	 *
	 * @return 构造时传入的 {@link CaptchaConfig}
	 */
	@Getter
	private final CaptchaConfig config;
	private final OrtSession.SessionOptions sessionOptions;

	/**
	 * 构造门面，校验配置并初始化底层识别器。
	 *
	 * @param config 验证码识别配置
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public CaptchaEngine(CaptchaConfig config) {
		Objects.requireNonNull(config, "CaptchaConfig must not be null");
		config.validate();
		this.config = config;
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = OrtSessionFactory.build(config.getOnnx());
		this.detector = new CaptchaDetector(environment, config, sessionOptions);
		log.info("mica-ai-captcha 初始化完成: model={} height={} normalization={}",
			config.getModelPath(), config.getInputHeight(), config.getNormalization());
	}

	/**
	 * 创建 {@link CaptchaEngine} 实例。
	 *
	 * @param config 验证码识别配置
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public static CaptchaEngine create(CaptchaConfig config) {
		return new CaptchaEngine(config);
	}

	/**
	 * 识别验证码图片（字节入参）。
	 *
	 * @param imageBytes 图片字节（png / jpg 等）
	 * @return 识别文本
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public String recognize(byte[] imageBytes) {
		return recognizeDetailed(imageBytes).getText();
	}

	/**
	 * 识别验证码图片（base64 入参，自动剥离 {@code data:image/png;base64,} 前缀）。
	 *
	 * @param base64Image base64 图片字符串
	 * @return 识别文本
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public String recognizeBase64(String base64Image) {
		return recognizeDetailedBase64(base64Image).getText();
	}

	/**
	 * 识别验证码图片（文件路径入参）。
	 *
	 * @param imagePath 图片文件路径
	 * @return 识别文本
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public String recognizePath(String imagePath) {
		return recognizeDetailedPath(imagePath).getText();
	}

	/**
	 * 识别验证码图片，返回带置信度的完整结果（字节入参）。
	 *
	 * @param imageBytes 图片字节（png / jpg 等）
	 * @return 识别结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public CaptchaResult recognizeDetailed(byte[] imageBytes) {
		return detector.recognize(CaptchaImageUtils.loadImage(imageBytes));
	}

	/**
	 * 识别验证码图片，返回带置信度的完整结果（base64 入参）。
	 *
	 * @param base64Image base64 图片字符串
	 * @return 识别结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public CaptchaResult recognizeDetailedBase64(String base64Image) {
		return detector.recognize(CaptchaImageUtils.loadBase64Image(base64Image));
	}

	/**
	 * 识别验证码图片，返回带置信度的完整结果（文件路径入参）。
	 *
	 * @param imagePath 图片文件路径
	 * @return 识别结果
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public CaptchaResult recognizeDetailedPath(String imagePath) {
		return detector.recognize(CaptchaImageUtils.loadImage(imagePath));
	}

	/**
	 * 识别验证码图片，返回带置信度的完整结果（{@link BufferedImage} 入参）。
	 *
	 * @param image 待识别图片
	 * @return 识别结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public CaptchaResult recognizeDetailed(BufferedImage image) {
		return detector.recognize(image);
	}

	/**
	 * 模型输入高度（NCHW 倒数第二维）。
	 *
	 * @return 输入高度；读取失败返回 -1
	 */
	public int modelInputHeight() {
		return detector.modelInputHeight();
	}

	/**
	 * 字符集类别数（含 blank），应与模型输出类别数一致。
	 *
	 * @return 类别数
	 */
	public int charsetSize() {
		return detector.charsetSize();
	}

	/**
	 * 释放底层 ONNX session 与会话选项，可重复调用。
	 */
	@Override
	public void close() {
		try {
			if (detector != null) {
				detector.close();
			}
		} finally {
			try {
				if (sessionOptions != null) {
					sessionOptions.close();
				}
			} catch (Exception e) {
				log.warn("mica-ai-captcha: 关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
			}
		}
	}
}
