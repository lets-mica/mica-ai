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
package net.dreamlu.mica.ai.docalign.core;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.docalign.config.DocAlignConfig;
import net.dreamlu.mica.ai.docalign.detection.DocAlignDetector;
import net.dreamlu.mica.ai.docalign.model.DocAlignResult;
import net.dreamlu.mica.ai.docalign.util.DocAlignImageUtils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * DocAligner 文档角点检测 / 透视矫正门面。
 *
 * <p>典型用法：
 * <pre>{@code
 * DocAlignConfig config = DocAlignConfig.builder()
 *     .modelPath("model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx")
 *     .build();
 * try (DocAlignEngine engine = DocAlignEngine.create(config)) {
 *     DocAlignResult corners = engine.cornersPath("photo.jpg");
 *     byte[] flat = engine.alignPath("photo.jpg");   // 拉平成矩形（PNG）
 * }
 * }</pre>
 *
 * <p>线程安全：内部 detector + ONNX session 线程安全；本类自身仅含构造时冻结的不可变
 * 字段，可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class DocAlignEngine implements AutoCloseable {

	private final DocAlignDetector detector;
	private final DocAlignConfig config;
	private final OrtSession.SessionOptions sessionOptions;

	/**
	 * 构造门面，校验配置并初始化底层检测器。
	 *
	 * @param config 角点检测配置
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public DocAlignEngine(DocAlignConfig config) {
		Objects.requireNonNull(config, "DocAlignConfig must not be null");
		config.validate();
		this.config = config;
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = OrtSessionFactory.build(config.getOnnx());
		this.detector = new DocAlignDetector(environment, config, sessionOptions);
		log.info("mica-ai-docalign 初始化完成: version={} input={}x{} threshold={}",
			config.getModelVersion(), config.getInputWidth(), config.getInputHeight(),
			config.getHeatmapThreshold());
	}

	/**
	 * 创建 {@link DocAlignEngine} 实例。
	 *
	 * @param config 角点检测配置
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public static DocAlignEngine create(DocAlignConfig config) {
		return new DocAlignEngine(config);
	}

	/**
	 * 检测文档四角。
	 *
	 * @param bgr 原图 BGR Mat
	 * @return 4 个角点（左上 / 右上 / 右下 / 左下）；输入为 null / 空图时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public DocAlignResult corners(Mat bgr) {
		return detector.corners(bgr);
	}

	/**
	 * 检测文档四角（字节入参）。
	 *
	 * @param imageBytes 图像字节
	 * @return 4 个角点
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public DocAlignResult cornersBytes(byte[] imageBytes) {
		return detector.cornersBytes(imageBytes);
	}

	/**
	 * 检测文档四角（文件路径入参）。
	 *
	 * @param imagePath 图像文件路径
	 * @return 4 个角点
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public DocAlignResult cornersPath(String imagePath) {
		return detector.cornersBytes(readFile(imagePath));
	}

	/**
	 * 按角点做透视矫正，输出尺寸由角点边长自动推算。
	 *
	 * @param bgr     原图 BGR Mat
	 * @param corners 角点检测结果
	 * @return 拉平后的 BGR Mat，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 输出尺寸非法
	 */
	public Mat warp(Mat bgr, DocAlignResult corners) {
		int[] size = DocAlignImageUtils.suggestOutputSize(corners);
		return DocAlignImageUtils.warp(bgr, corners, size[0], size[1]);
	}

	/**
	 * 按角点做透视矫正到指定尺寸。
	 *
	 * @param bgr       原图 BGR Mat
	 * @param corners   角点检测结果
	 * @param outWidth  输出宽度
	 * @param outHeight 输出高度
	 * @return 拉平后的 BGR Mat，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 输出尺寸非法
	 */
	public Mat warp(Mat bgr, DocAlignResult corners, int outWidth, int outHeight) {
		return DocAlignImageUtils.warp(bgr, corners, outWidth, outHeight);
	}

	/**
	 * 一步完成「检测角点 + 透视矫正」，输出拉平后的 PNG 字节。
	 *
	 * @param imageBytes 图像字节
	 * @return PNG 字节
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败；
	 *                         {@link ErrorCode#ENCODE_FAILED} 编码失败
	 */
	public byte[] alignBytes(byte[] imageBytes) {
		Mat bgr = null;
		Mat warped = null;
		try {
			bgr = DocAlignImageUtils.byteArrayToMat(imageBytes);
			DocAlignResult corners = detector.corners(bgr);
			int[] size = DocAlignImageUtils.suggestOutputSize(corners);
			warped = DocAlignImageUtils.warp(bgr, corners, size[0], size[1]);
			return DocAlignImageUtils.imencodePng(warped);
		} finally {
			DocAlignImageUtils.releaseAll(bgr, warped);
		}
	}

	/**
	 * 一步完成「检测角点 + 透视矫正」（文件路径入参）。
	 *
	 * @param imagePath 图像文件路径
	 * @return PNG 字节
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public byte[] alignPath(String imagePath) {
		return alignBytes(readFile(imagePath));
	}

	/**
	 * 获取当前配置。
	 *
	 * @return 构造时传入的 {@link DocAlignConfig}
	 */
	public DocAlignConfig getConfig() {
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
	 * 模型输出节点个数（DocAligner 应为 1）。
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
				log.warn("mica-ai-docalign: 关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
			}
		}
	}
}
