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
package net.dreamlu.mica.ai.textline;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.textline.config.TextLineConfig;
import net.dreamlu.mica.ai.textline.detection.TextLineDetector;
import net.dreamlu.mica.ai.textline.model.TextLineOrientationResult;
import net.dreamlu.mica.ai.textline.util.TextLineImageUtils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * 文本行方向分类门面。
 *
 * <p>对外提供三种粒度的能力：
 * <ul>
 *   <li>{@link #classify(Mat)} / {@link #classifyBytes(byte[])} / {@link #classifyPath(String)} —
 *       只判定方向，返回 {@link TextLineOrientationResult}</li>
 *   <li>{@link #uprightBytes(byte[])} — 若判定为倒置则旋转 180 度并输出 PNG，
 *       否则<b>原样返回</b>输入字节；适合「无脑转正」的流水线</li>
 *   <li>{@link #rotateIfUpsideDown(Mat)} — 返回转正后的 Mat（可能与原图同一引用语义由返回值决定），
 *       供自行续接后续处理</li>
 * </ul>
 *
 * <pre>{@code
 * TextLineConfig config = TextLineConfig.builder()
 *     .modelPath("model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx")
 *     .build();
 * try (TextLineEngine engine = TextLineEngine.create(config)) {
 *     TextLineOrientationResult r = engine.classifyBytes(lineBytes);
 *     if (r.isUpsideDown()) {
 *         System.out.println("该行倒置，置信度 " + r.getScore());
 *     }
 *
 *     // 或者一把梭：倒置则转正，正常则原样返回
 *     byte[] upright = engine.uprightBytes(lineBytes);
 * }
 * }</pre>
 *
 * <p>线程安全：内部 detector + ONNX session 线程安全；本类自身仅含构造时冻结的不可变
 * 字段，可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class TextLineEngine implements AutoCloseable {

	private final TextLineDetector detector;
	private final TextLineConfig config;
	private final OrtSession.SessionOptions sessionOptions;

	/**
	 * 构造门面，校验配置并初始化底层检测器。
	 *
	 * @param config 方向分类配置
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public TextLineEngine(TextLineConfig config) {
		Objects.requireNonNull(config, "TextLineConfig must not be null");
		config.validate();
		this.config = config;
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = OrtSessionFactory.build(config.getOnnx());
		this.detector = new TextLineDetector(environment, config, sessionOptions);
		log.info("mica-ai-textline 初始化完成: version={} input={}x{} threshold={}",
			config.getModelVersion(), config.getInputWidth(), config.getInputHeight(),
			config.getUpsideDownThreshold());
	}

	/**
	 * 创建 {@link TextLineEngine} 实例。
	 *
	 * @param config 方向分类配置
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public static TextLineEngine create(TextLineConfig config) {
		return new TextLineEngine(config);
	}

	/**
	 * 使用默认配置创建 {@link TextLineEngine} 实例。
	 *
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载失败
	 */
	public static TextLineEngine createDefault() {
		return new TextLineEngine(TextLineConfig.builder().build());
	}

	/**
	 * 判定文本行方向。
	 *
	 * @param line 文本行图 BGR Mat
	 * @return 判定结果；输入为 null / 空图时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public TextLineOrientationResult classify(Mat line) {
		return detector.classify(line);
	}

	/**
	 * 判定文本行方向（字节入参）。
	 *
	 * @param imageBytes 文本行图字节（png / jpg 等）
	 * @return 判定结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public TextLineOrientationResult classifyBytes(byte[] imageBytes) {
		return detector.classifyBytes(imageBytes);
	}

	/**
	 * 判定文本行方向（文件路径入参）。
	 *
	 * @param imagePath 文本行图文件路径
	 * @return 判定结果
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public TextLineOrientationResult classifyPath(String imagePath) {
		return detector.classifyBytes(readFile(imagePath));
	}

	/**
	 * 若文本行倒置则旋转 180 度，否则返回原图。
	 *
	 * <p>⚠️ <b>返回值可能是入参本身</b>（方向正常时）。调用方若要长期持有，应自行复制；
	 * 若用 try-with-resources 风格释放，注意不要重复释放入参。
	 *
	 * @param line 文本行图 BGR Mat
	 * @return 转正后的 Mat；输入为 null / 空图时返回 null；
	 *         方向正常时返回<b>入参本身</b>
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public Mat rotateIfUpsideDown(Mat line) {
		TextLineOrientationResult result = detector.classify(line);
		if (result == null || !result.isUpsideDown()) {
			return line;
		}
		return TextLineImageUtils.rotate180(line);
	}

	/**
	 * 「无脑转正」便捷方法：判定为倒置则旋转 180 度并输出 PNG 字节；
	 * 方向正常时<b>原样返回输入字节</b>（不做无谓的重新编码，避免画质损失）。
	 *
	 * @param imageBytes 文本行图字节
	 * @return 转正后的 PNG 字节，或原输入字节
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败；
	 *                         {@link ErrorCode#ENCODE_FAILED} 编码失败
	 */
	public byte[] uprightBytes(byte[] imageBytes) {
		Mat bgr = null;
		Mat rotated = null;
		try {
			bgr = TextLineImageUtils.byteArrayToMat(imageBytes);
			TextLineOrientationResult result = detector.classify(bgr);
			if (result == null || !result.isUpsideDown()) {
				return imageBytes;
			}
			rotated = TextLineImageUtils.rotate180(bgr);
			return TextLineImageUtils.imencodePng(rotated);
		} finally {
			TextLineImageUtils.releaseAll(bgr, rotated);
		}
	}

	/**
	 * 获取当前配置。
	 *
	 * @return 构造时传入的 {@link TextLineConfig}
	 */
	public TextLineConfig getConfig() {
		return config;
	}

	/**
	 * 模型输入宽度（NCHW 最后一维）。
	 *
	 * @return 输入宽度；读取失败返回 -1
	 */
	public int modelInputWidth() {
		return detector.modelInputWidth();
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
	 * 模型分类数（官方模型应为 2）。
	 *
	 * @return 分类数；读取失败返回 -1
	 */
	public int classCount() {
		return detector.classCount();
	}

	private static byte[] readFile(String imagePath) {
		Path p = Paths.get(imagePath);
		if (!Files.exists(p)) {
			throw new MicaAiException(ErrorCode.NOT_FOUND, "文件不存在: " + imagePath);
		}
		try {
			return Files.readAllBytes(p);
		} catch (IOException e) {
			throw new MicaAiException(ErrorCode.UNKNOWN, "读取图像失败: " + imagePath, e);
		}
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
				log.warn("mica-ai-textline: 关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
			}
		}
	}
}
