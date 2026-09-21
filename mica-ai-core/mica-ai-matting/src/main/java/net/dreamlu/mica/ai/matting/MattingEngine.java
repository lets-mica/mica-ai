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
package net.dreamlu.mica.ai.matting;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.matting.config.MattingConfig;
import net.dreamlu.mica.ai.matting.detection.MattingDetector;
import net.dreamlu.mica.ai.matting.model.MattingResult;
import net.dreamlu.mica.ai.matting.util.MattingImageUtils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * u2netp 通用抠图门面。
 *
 * <p>对外提供三种粒度的能力：
 * <ul>
 *   <li>{@link #alpha(Mat)} / {@link #alphaBytes(byte[])} / {@link #alphaPath(String)} —
 *       只要原尺寸 alpha 掩码，自行决定后续合成</li>
 *   <li>{@link #cutoutBytes(byte[])} / {@link #cutoutPath(String)} —
 *       直接输出 <b>透明底 PNG</b>（BGRA）字节</li>
 *   <li>{@link #cutoutOnColorBytes(byte[], int[])} — 输出指定纯色底的 PNG/JPEG 字节</li>
 * </ul>
 *
 * <pre>{@code
 * MattingConfig config = MattingConfig.builder()
 *     .modelPath("model-tools/matting/models/u2netp.onnx")
 *     .build();
 * try (MattingEngine engine = MattingEngine.create(config)) {
 *     byte[] png = engine.cutoutBytes(Files.readAllBytes(Paths.get("photo.jpg")));
 *     Files.write(Paths.get("photo-nobg.png"), png);
 *
 *     // 或者只要 alpha
 *     try (MattingResult r = engine.matteBytes(Files.readAllBytes(Paths.get("photo.jpg")))) {
 *         Mat alpha = r.getAlpha();
 *     }
 * }
 * }</pre>
 *
 * <p>线程安全：内部 detector + ONNX session 线程安全；本类自身仅含构造时冻结的不可变
 * 字段，可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class MattingEngine implements AutoCloseable {

	private final MattingDetector detector;
	private final MattingConfig config;
	private final OrtSession.SessionOptions sessionOptions;

	/**
	 * 构造门面，校验配置并初始化底层检测器。
	 *
	 * @param config 抠图配置
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public MattingEngine(MattingConfig config) {
		Objects.requireNonNull(config, "MattingConfig must not be null");
		config.validate();
		this.config = config;
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = OrtSessionFactory.build(config.getOnnx());
		this.detector = new MattingDetector(environment, config, sessionOptions);
		log.info("mica-ai-matting 初始化完成: version={} inputSize={} normalize={} threshold={}",
			config.getModelVersion(), config.getInputSize(),
			config.isMinMaxNormalize(), config.getBinaryThreshold());
	}

	/**
	 * 创建 {@link MattingEngine} 实例。
	 *
	 * @param config 抠图配置
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public static MattingEngine create(MattingConfig config) {
		return new MattingEngine(config);
	}

	/**
	 * 使用默认配置创建 {@link MattingEngine} 实例。
	 *
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载失败
	 */
	public static MattingEngine createDefault() {
		return new MattingEngine(MattingConfig.builder().build());
	}

	/**
	 * 计算原尺寸 alpha 掩码。
	 *
	 * @param bgr 原图 BGR Mat
	 * @return 原尺寸单通道 {@code CV_32FC1} 掩码，调用方负责 release；输入为 null / 空图时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public Mat alpha(Mat bgr) {
		return detector.alpha(bgr);
	}

	/**
	 * 计算原尺寸 alpha 掩码（字节入参）。
	 *
	 * @param imageBytes 图像字节（png / jpg 等）
	 * @return 原尺寸单通道 {@code CV_32FC1} 掩码，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public Mat alphaBytes(byte[] imageBytes) {
		return detector.alphaBytes(imageBytes);
	}

	/**
	 * 计算原尺寸 alpha 掩码（文件路径入参）。
	 *
	 * @param imagePath 图像文件路径
	 * @return 原尺寸单通道 {@code CV_32FC1} 掩码，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public Mat alphaPath(String imagePath) {
		byte[] bytes = readFile(imagePath);
		return detector.alphaBytes(bytes);
	}

	/**
	 * 抠图并返回带 alpha 的结果对象（便于自行合成）。
	 *
	 * @param bgr 原图 BGR Mat
	 * @return 抠图结果；输入为 null / 空图时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public MattingResult matte(Mat bgr) {
		Mat alpha = detector.alpha(bgr);
		if (alpha == null) {
			return null;
		}
		return new MattingResult(alpha, bgr.cols(), bgr.rows());
	}

	/**
	 * 抠图（字节入参）。
	 *
	 * @param imageBytes 图像字节
	 * @return 抠图结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public MattingResult matteBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = MattingImageUtils.byteArrayToMat(imageBytes);
			return matte(bgr);
		} finally {
			MattingImageUtils.releaseAll(bgr);
		}
	}

	/**
	 * 抠图并输出<b>透明底 PNG</b>（BGRA）字节。
	 *
	 * @param imageBytes 图像字节
	 * @return PNG 字节
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败；
	 *                         {@link ErrorCode#ENCODE_FAILED} 编码失败
	 */
	public byte[] cutoutBytes(byte[] imageBytes) {
		Mat bgr = null;
		Mat alpha = null;
		Mat bgra = null;
		try {
			bgr = MattingImageUtils.byteArrayToMat(imageBytes);
			alpha = detector.alpha(bgr);
			bgra = MattingImageUtils.toBgra(bgr, alpha);
			return MattingImageUtils.imencodePng(bgra);
		} finally {
			MattingImageUtils.releaseAll(bgr, alpha, bgra);
		}
	}

	/**
	 * 抠图并输出<b>透明底 PNG</b>（BGRA）字节（文件路径入参）。
	 *
	 * @param imagePath 图像文件路径
	 * @return PNG 字节
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public byte[] cutoutPath(String imagePath) {
		return cutoutBytes(readFile(imagePath));
	}

	/**
	 * 抠图并把主体合成到纯色底上，输出 PNG 字节。
	 *
	 * @param imageBytes      图像字节
	 * @param backgroundColor 底色（RGB，0~255）；{@code null} 时取配置值
	 * @return PNG 字节
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public byte[] cutoutOnColorBytes(byte[] imageBytes, int[] backgroundColor) {
		int[] bg = backgroundColor == null ? config.getBackgroundColor() : backgroundColor;
		if (bg.length != 3) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"backgroundColor 必须为长度 3 的数组");
		}
		Mat bgr = null;
		Mat alpha = null;
		Mat composed = null;
		try {
			bgr = MattingImageUtils.byteArrayToMat(imageBytes);
			alpha = detector.alpha(bgr);
			composed = MattingImageUtils.compositeOnColor(bgr, alpha, bg);
			return MattingImageUtils.imencodePng(composed);
		} finally {
			MattingImageUtils.releaseAll(bgr, alpha, composed);
		}
	}

	/**
	 * 抠图并输出二值掩码字节（{@code 0/255} 单通道 PNG），适合需要硬边缘的场景。
	 *
	 * @param imageBytes 图像字节
	 * @return PNG 字节
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public byte[] matteBinaryBytes(byte[] imageBytes) {
		Mat bgr = null;
		Mat alpha = null;
		Mat binary = null;
		try {
			bgr = MattingImageUtils.byteArrayToMat(imageBytes);
			alpha = detector.alpha(bgr);
			binary = MattingImageUtils.threshold(alpha, config.getBinaryThreshold());
			return MattingImageUtils.imencodePng(binary);
		} finally {
			MattingImageUtils.releaseAll(bgr, alpha, binary);
		}
	}

	/**
	 * 获取当前配置。
	 *
	 * @return 构造时传入的 {@link MattingConfig}
	 */
	public MattingConfig getConfig() {
		return config;
	}

	/**
	 * 模型输入边长（NCHW 的最后两维）。
	 *
	 * @return 输入边长；读取失败返回 -1
	 */
	public int modelInputSize() {
		return detector.modelInputSize();
	}

	/**
	 * 模型输出节点个数（u2netp 应为 7，即 d0..d6）。
	 *
	 * @return 输出节点个数；读取失败返回 -1
	 */
	public int outputCount() {
		return detector.outputCount();
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
				log.warn("mica-ai-matting: 关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
			}
		}
	}
}
