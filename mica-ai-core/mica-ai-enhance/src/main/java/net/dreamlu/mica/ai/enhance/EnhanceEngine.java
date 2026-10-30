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
package net.dreamlu.mica.ai.enhance;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.enhance.config.EnhanceConfig;
import net.dreamlu.mica.ai.enhance.detection.EnhanceDetector;
import net.dreamlu.mica.ai.enhance.util.EnhanceImageUtils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * 图像超分门面（Real-ESRGAN realesr-general-x4v3）。
 *
 * <p>大图按 {@link EnhanceConfig#getTileSize()} 切 tile 推理：每个 tile 在四周多取
 * {@code tileOverlap / 2} 像素上下文、只回写自己的核心区域，从而在不受内存爆炸影响的同时
 * 消除拼接接缝。图像本身不超过一个 tile 时整图推理。
 *
 * <pre>{@code
 * EnhanceConfig config = EnhanceConfig.builder()
 *     .modelPath("model-tools/enhance/models/realesr-general-x4v3.onnx")
 *     .build();
 * try (EnhanceEngine engine = EnhanceEngine.create(config)) {
 *     byte[] png = engine.upscalePath("photo.jpg");   // 4 倍放大后的 PNG
 * }
 * }</pre>
 *
 * <p>线程安全：内部 detector + ONNX session 线程安全；本类自身仅含构造时冻结的不可变
 * 字段，可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class EnhanceEngine implements AutoCloseable {

	private final EnhanceDetector detector;
	@Getter
	private final EnhanceConfig config;
	private final OrtSession.SessionOptions sessionOptions;

	/**
	 * 构造门面，校验配置并初始化底层检测器。
	 *
	 * @param config 超分配置
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public EnhanceEngine(EnhanceConfig config) {
		Objects.requireNonNull(config, "EnhanceConfig must not be null");
		config.validate();
		this.config = config;
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = OrtSessionFactory.build(config.getOnnx());
		this.detector = new EnhanceDetector(environment, config, sessionOptions);
	}

	/**
	 * 创建 {@link EnhanceEngine} 实例。
	 *
	 * @param config 超分配置
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public static EnhanceEngine create(EnhanceConfig config) {
		return new EnhanceEngine(config);
	}

	/**
	 * 对 BGR Mat 做 {@code scale} 倍超分（自动 tile 切分与拼接）。
	 *
	 * @param bgr 原图 BGR Mat；null 或空图时返回 null
	 * @return {@code scale} 倍尺寸的 BGR Mat，调用方负责 release；输入为空时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public Mat upscale(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return null;
		}
		int width = bgr.cols();
		int height = bgr.rows();
		int scale = config.getScale();
		Mat dst = new Mat(height * scale, width * scale, CvType.CV_8UC3);
		int tileSize = config.getTileSize();
		int overlap = config.getTileOverlap();

		if (width <= tileSize && height <= tileSize) {
			Mat out = null;
			try {
				out = detector.upscale(bgr);
				out.copyTo(dst);
				return dst;
			} finally {
				EnhanceImageUtils.releaseAll(out);
			}
		}

		int step = Math.max(1, tileSize - overlap);
		int margin = overlap / 2;
		for (int y = 0; y < height; y += step) {
			int coreH = Math.min(step, height - y);
			for (int x = 0; x < width; x += step) {
				int coreW = Math.min(step, width - x);
				int inX0 = Math.max(0, x - margin);
				int inY0 = Math.max(0, y - margin);
				int inX1 = Math.min(width, x + coreW + margin);
				int inY1 = Math.min(height, y + coreH + margin);
				Mat tile = null;
				Mat out = null;
				Mat srcRoi = null;
				Mat dstRoi = null;
				try {
					tile = EnhanceImageUtils.crop(bgr, inX0, inY0,
						inX1 - inX0, inY1 - inY0);
					out = detector.upscale(tile);
					int offX = (x - inX0) * scale;
					int offY = (y - inY0) * scale;
					srcRoi = new Mat(out, new Rect(offX, offY,
						coreW * scale, coreH * scale));
					dstRoi = new Mat(dst, new Rect(x * scale, y * scale,
						coreW * scale, coreH * scale));
					srcRoi.copyTo(dstRoi);
				} finally {
					EnhanceImageUtils.releaseAll(srcRoi, dstRoi, tile, out);
				}
			}
		}
		return dst;
	}

	/**
	 * 解码字节数组为 BGR Mat 并做超分，返回 PNG 字节。
	 *
	 * @param imageBytes 图像字节（非空）
	 * @return {@code scale} 倍尺寸的 PNG 字节
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 * @throws MicaAiException {@link ErrorCode#ENCODE_FAILED} PNG 编码失败
	 */
	public byte[] upscaleBytes(byte[] imageBytes) {
		Mat bgr = null;
		Mat out = null;
		try {
			bgr = EnhanceImageUtils.byteArrayToMat(imageBytes);
			out = upscale(bgr);
			return EnhanceImageUtils.imencodePng(out);
		} finally {
			EnhanceImageUtils.releaseAll(bgr, out);
		}
	}

	/**
	 * 读取图片文件并做超分，返回 PNG 字节。
	 *
	 * @param imagePath 图片文件路径
	 * @return {@code scale} 倍尺寸的 PNG 字节
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 文件读取失败
	 * @throws MicaAiException {@link ErrorCode#ENCODE_FAILED} PNG 编码失败
	 */
	public byte[] upscalePath(String imagePath) {
		Objects.requireNonNull(imagePath, "imagePath must not be null");
		Path path = Paths.get(imagePath);
		try {
			return upscaleBytes(Files.readAllBytes(path));
		} catch (IOException e) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
				"读取图片失败: " + imagePath, e);
		}
	}

	/**
	 * 便于测试断言：模型输出节点数（应为 1）。
	 *
	 * @return 输出节点个数
	 */
	public int outputCount() {
		return detector.outputCount();
	}

	/**
	 * 释放底层 ONNX session 与会话选项。
	 */
	@Override
	public void close() {
		if (detector != null) {
			detector.close();
		}
		if (sessionOptions != null) {
			sessionOptions.close();
		}
	}
}
