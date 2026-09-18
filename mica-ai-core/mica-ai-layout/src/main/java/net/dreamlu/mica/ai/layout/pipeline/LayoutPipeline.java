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
package net.dreamlu.mica.ai.layout.pipeline;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.layout.config.LayoutConfig;
import net.dreamlu.mica.ai.layout.detection.LayoutDetector;
import net.dreamlu.mica.ai.layout.model.LayoutResult;
import net.dreamlu.mica.ai.layout.util.LayoutImageUtils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * PP-DocLayoutV2 / V3 文档版面分析门面。
 *
 * <p>对外提供 {@code detectPath / detectBytes / detect(Mat)} 三种入参；统一通过
 * {@link LayoutDetector} 走 ONNX 推理 + 反 letterbox + NMS 后处理 + 阅读顺序解码。
 * {@link AutoCloseable} 实现保证底层 {@code OrtSession} 在作用域结束时被正确关闭。
 *
 * <pre>{@code
 * LayoutConfig config = LayoutConfig.builder()
 *     .modelPath("model-tools/layout/models/model.onnx")
 *     .build();
 * try (LayoutPipeline pipeline = LayoutPipeline.create(config)) {
 *     List<LayoutResult> regions = pipeline.detectPath("doc.png");
 *     regions.forEach(r -> System.out.println(
 *         r.getLabelCode() + " " + r.getScore() + " order=" + r.getReadingOrder()));
 * }
 * }</pre>
 *
 * <p>线程安全：内部 detector + ONNX session 线程安全；本类自身仅含构造时冻结的不可变
 * 字段，可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class LayoutPipeline implements AutoCloseable {

	private final LayoutDetector detector;
	private final LayoutConfig config;
	private final OrtSession.SessionOptions sessionOptions;

	/**
	 * 构造门面，校验配置并初始化底层检测器。
	 *
	 * @param config 版面分析配置
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public LayoutPipeline(LayoutConfig config) {
		Objects.requireNonNull(config, "LayoutConfig must not be null");
		config.validate();
		this.config = config;
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = buildSessionOptions(config.getOnnx());
		this.detector = new LayoutDetector(environment, config, sessionOptions);
		log.info("mica-ai-layout 初始化完成: version={} maxSide={} score={} nms={} maxDet={}",
			config.getModelVersion(), config.getMaxSideLength(),
			config.getScoreThreshold(), config.isLayoutNms() ? config.getNmsThreshold() : "off",
			config.getMaxDetections());
	}

	/**
	 * 创建 {@link LayoutPipeline} 实例。
	 *
	 * @param config 版面分析配置
	 * @return 初始化完成的 pipeline
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public static LayoutPipeline create(LayoutConfig config) {
		return new LayoutPipeline(config);
	}

	/**
	 * 使用默认配置创建 {@link LayoutPipeline} 实例。
	 *
	 * @return 初始化完成的 pipeline
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 模型加载失败
	 */
	public static LayoutPipeline createDefault() {
		return new LayoutPipeline(LayoutConfig.builder().build());
	}

	private static OrtSession.SessionOptions buildSessionOptions(OrtSessionOptions onnx) {
		OrtSession.SessionOptions so = new OrtSession.SessionOptions();
		try {
			if (onnx != null) {
				if (onnx.getIntraOpNumThreads() > 0) {
					so.setIntraOpNumThreads(onnx.getIntraOpNumThreads());
				}
				if (onnx.getInterOpNumThreads() > 0) {
					so.setInterOpNumThreads(onnx.getInterOpNumThreads());
				}
				OrtDevice device = onnx.getDevice();
				if (OrtDevice.GPU == device) {
					Set<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
					if (providers != null && providers.contains(OrtProvider.CUDA)) {
						so.addCUDA(onnx.getCudaDeviceId());
						log.info("mica-ai-layout: 已启用 CUDA 执行提供器 (deviceId={})", onnx.getCudaDeviceId());
					} else {
						log.warn("mica-ai-layout: 未检测到可用的 CUDA 执行提供器，回退到 CPU 推理");
					}
				}
			}
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "配置 ONNX 会话选项失败", e);
		}
		return so;
	}

	/**
	 * 检测本地图像文件的版面区域。
	 *
	 * @param imagePath 图像文件路径
	 * @return 按阅读顺序排列的版面区域列表；文件不存在时抛异常
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public List<LayoutResult> detectPath(String imagePath) {
		Path p = Paths.get(imagePath);
		if (!Files.exists(p)) {
			throw new MicaAiException(ErrorCode.NOT_FOUND, "文件不存在: " + imagePath);
		}
		Mat bgr = null;
		try {
			bgr = LayoutImageUtils.byteArrayToMat(Files.readAllBytes(p));
			return detect(bgr);
		} catch (IOException e) {
			throw new MicaAiException(ErrorCode.UNKNOWN, "读取图像失败: " + imagePath, e);
		} finally {
			LayoutImageUtils.releaseAll(bgr);
		}
	}

	/**
	 * 检测图像字节数组的版面区域。
	 *
	 * @param imageBytes 图像文件字节（png / jpg 等）
	 * @return 按阅读顺序排列的版面区域列表
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public List<LayoutResult> detectBytes(byte[] imageBytes) {
		return detector.detectBytes(imageBytes);
	}

	/**
	 * 检测 BGR {@link Mat} 图像的版面区域。
	 *
	 * @param bgr 原图 BGR Mat；null 或空图时返回空列表
	 * @return 按阅读顺序排列的版面区域列表
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public List<LayoutResult> detect(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return Collections.emptyList();
		}
		return detector.detect(bgr);
	}

	/**
	 * 获取当前配置。
	 *
	 * @return 构造时传入的 {@link LayoutConfig}
	 */
	public LayoutConfig getConfig() {
		return config;
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
				log.warn("mica-ai-layout: 关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
			}
		}
	}
}