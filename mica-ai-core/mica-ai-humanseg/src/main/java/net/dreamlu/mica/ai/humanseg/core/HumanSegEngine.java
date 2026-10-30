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
package net.dreamlu.mica.ai.humanseg.core;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.humanseg.config.HumanSegConfig;
import net.dreamlu.mica.ai.humanseg.detection.HumanSegDetector;
import net.dreamlu.mica.ai.humanseg.model.HumanSegResult;
import net.dreamlu.mica.ai.humanseg.util.HumanSegImageUtils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * PP-HumanSeg 人像分割门面。
 *
 * <p>对外提供三种粒度：
 * <ul>
 *   <li>{@link #alpha(Mat)} / {@link #alphaBytes(byte[])} / {@link #alphaPath(String)} —
 *       只要原尺寸 alpha 掩码，自行决定后续合成</li>
 *   <li>{@link #cutoutBytes(byte[])} / {@link #cutoutPath(String)} / {@link #cutoutOnColorBytes(byte[], int[])} —
 *       直接输出透明底或纯色底 PNG</li>
 *   <li>{@link #maskBinaryBytes(byte[])} — 二值掩码（{@code 0/255} 单通道 PNG）</li>
 * </ul>
 *
 * <pre>{@code
 * HumanSegConfig config = HumanSegConfig.builder()
 *     .modelPath("model-tools/humanseg/models/human_segmentation_pphumanseg_2023mar.onnx")
 *     .build();
 * try (HumanSegEngine engine = HumanSegEngine.create(config)) {
 *     byte[] png = engine.cutoutPath("portrait.jpg");
 * }
 * }</pre>
 *
 * <p>线程安全：内部 detector + ONNX session 线程安全；本类自身仅含构造时冻结的不可变
 * 字段，可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class HumanSegEngine implements AutoCloseable {

	private final HumanSegDetector detector;
	private final HumanSegConfig config;
	private final OrtSession.SessionOptions sessionOptions;

	/**
	 * 构造门面，校验配置并初始化底层检测器。
	 *
	 * @param config 人像分割配置
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public HumanSegEngine(HumanSegConfig config) {
		Objects.requireNonNull(config, "HumanSegConfig must not be null");
		config.validate();
		this.config = config;
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = OrtSessionFactory.build(config.getOnnx());
		this.detector = new HumanSegDetector(environment, config, sessionOptions);
		log.info("mica-ai-humanseg 初始化完成: version={} input={}x{} threshold={}",
			config.getModelVersion(), config.getInputWidth(), config.getInputHeight(),
			config.getBinaryThreshold());
	}

	/**
	 * 创建 {@link HumanSegEngine} 实例。
	 *
	 * @param config 人像分割配置
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public static HumanSegEngine create(HumanSegConfig config) {
		return new HumanSegEngine(config);
	}

	/**
	 * 使用默认配置（内置 classpath 模型）创建引擎。
	 *
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载失败
	 */
	public static HumanSegEngine createDefault() {
		return new HumanSegEngine(HumanSegConfig.builder().build());
	}

	/**
	 * 计算原尺寸人像 alpha 掩码。
	 *
	 * @param bgr 原图 BGR Mat
	 * @return 原尺寸单通道 {@code CV_32FC1} 掩码，调用方负责 release；输入为 null / 空图时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public Mat alpha(Mat bgr) {
		return detector.alpha(bgr);
	}

	/**
	 * 计算原尺寸人像 alpha 掩码（字节入参）。
	 *
	 * @param imageBytes 图像字节（png / jpg 等）
	 * @return 原尺寸单通道 {@code CV_32FC1} 掩码，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public Mat alphaBytes(byte[] imageBytes) {
		return detector.alphaBytes(imageBytes);
	}

	/**
	 * 计算原尺寸人像 alpha 掩码（文件路径入参）。
	 *
	 * @param imagePath 图像文件路径
	 * @return 原尺寸单通道 {@code CV_32FC1} 掩码，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public Mat alphaPath(String imagePath) {
		return detector.alphaBytes(readFile(imagePath));
	}

	/**
	 * 分割并返回带 alpha 的结果对象（便于自行合成）。
	 *
	 * @param bgr 原图 BGR Mat
	 * @return 分割结果；输入为 null / 空图时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public HumanSegResult matte(Mat bgr) {
		Mat alpha = detector.alpha(bgr);
		if (alpha == null) {
			return null;
		}
		return new HumanSegResult(alpha, bgr.cols(), bgr.rows());
	}

	/**
	 * 分割（字节入参）。
	 *
	 * @param imageBytes 图像字节
	 * @return 分割结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public HumanSegResult matteBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = HumanSegImageUtils.byteArrayToMat(imageBytes);
			return matte(bgr);
		} finally {
			HumanSegImageUtils.releaseAll(bgr);
		}
	}

	/**
	 * 分割并输出<b>透明底 PNG</b>（BGRA）字节。
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
			bgr = HumanSegImageUtils.byteArrayToMat(imageBytes);
			alpha = detector.alpha(bgr);
			bgra = HumanSegImageUtils.toBgra(bgr, alpha);
			return HumanSegImageUtils.imencodePng(bgra);
		} finally {
			HumanSegImageUtils.releaseAll(bgr, alpha, bgra);
		}
	}

	/**
	 * 分割并输出<b>透明底 PNG</b>（文件路径入参）。
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
	 * 分割并把人像合成到纯色底上，输出 PNG 字节。
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
			bgr = HumanSegImageUtils.byteArrayToMat(imageBytes);
			alpha = detector.alpha(bgr);
			composed = HumanSegImageUtils.compositeOnColor(bgr, alpha, bg);
			return HumanSegImageUtils.imencodePng(composed);
		} finally {
			HumanSegImageUtils.releaseAll(bgr, alpha, composed);
		}
	}

	/**
	 * 分割并输出二值掩码字节（{@code 0/255} 单通道 PNG），适合需要硬边缘的场景。
	 *
	 * @param imageBytes 图像字节
	 * @return PNG 字节
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public byte[] maskBinaryBytes(byte[] imageBytes) {
		Mat bgr = null;
		Mat alpha = null;
		Mat binary = null;
		try {
			bgr = HumanSegImageUtils.byteArrayToMat(imageBytes);
			alpha = detector.alpha(bgr);
			binary = HumanSegImageUtils.threshold(alpha, config.getBinaryThreshold());
			return HumanSegImageUtils.imencodePng(binary);
		} finally {
			HumanSegImageUtils.releaseAll(bgr, alpha, binary);
		}
	}

	/**
	 * 获取当前配置。
	 *
	 * @return 构造时传入的 {@link HumanSegConfig}
	 */
	public HumanSegConfig getConfig() {
		return config;
	}

	/**
	 * 模型输入宽度。
	 *
	 * @return 输入宽度；读取失败返回 -1
	 */
	public int modelInputWidth() {
		return detector.modelInputWidth();
	}

	/**
	 * 模型输入高度。
	 *
	 * @return 输入高度；读取失败返回 -1
	 */
	public int modelInputHeight() {
		return detector.modelInputHeight();
	}

	/**
	 * 模型输出节点个数（PP-HumanSeg 应为 1）。
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
				log.warn("mica-ai-humanseg: 关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
			}
		}
	}
}
