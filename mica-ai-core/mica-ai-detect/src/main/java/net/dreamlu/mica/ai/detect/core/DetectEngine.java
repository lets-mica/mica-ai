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
package net.dreamlu.mica.ai.detect.core;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.detect.config.DetectConfig;
import net.dreamlu.mica.ai.detect.detection.ObjectDetector;
import net.dreamlu.mica.ai.detect.model.CocoLabels;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import net.dreamlu.mica.ai.detect.util.DetectImageUtils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * 通用目标检测门面（NanoDet-Plus / YOLOX-s，COCO 80 类）。
 *
 * <pre>{@code
 * try (DetectEngine engine = DetectEngine.createDefault()) {
 *     DetectResult result = engine.detectPath("street.jpg");
 *     for (DetectBox box : result.getBoxes()) {
 *         System.out.println(box.label(engine.getLabels()) + " " + box.getScore());
 *     }
 * }
 * }</pre>
 *
 * <p>线程安全：内部 detector + ONNX session 线程安全；本类自身仅含构造时冻结的不可变
 * 字段，可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class DetectEngine implements AutoCloseable {

	private final ObjectDetector detector;
	private final DetectConfig config;
	private final OrtSession.SessionOptions sessionOptions;
	private final String[] labels;

	/**
	 * 构造门面，校验配置并初始化底层检测器。
	 *
	 * @param config 检测配置
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public DetectEngine(DetectConfig config) {
		Objects.requireNonNull(config, "DetectConfig must not be null");
		config.validate();
		this.config = config;
		this.labels = config.getLabels() == null ? CocoLabels.COCO_80 : config.getLabels();
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = OrtSessionFactory.build(config.getOnnx());
		this.detector = new ObjectDetector(environment, config, sessionOptions);
		log.info("mica-ai-detect 初始化完成: family={} classCount={}",
			config.getFamily(), this.labels.length);
	}

	/**
	 * 创建 {@link DetectEngine} 实例。
	 *
	 * @param config 检测配置
	 * @return 初始化完成的引擎
	 */
	public static DetectEngine create(DetectConfig config) {
		return new DetectEngine(config);
	}

	/**
	 * 使用默认配置（NanoDet-Plus，内置 classpath 模型）创建引擎。
	 *
	 * @return 初始化完成的引擎
	 */
	public static DetectEngine createDefault() {
		return new DetectEngine(DetectConfig.builder().build());
	}

	/**
	 * 检测单张图像。
	 *
	 * @param bgr 原图 BGR Mat；null 或空图时返回 null
	 * @return 检测结果；输入为空时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public DetectResult detect(Mat bgr) {
		return detector.detect(bgr);
	}

	/**
	 * 检测图像字节。
	 *
	 * @param imageBytes 图像字节（png / jpg 等）
	 * @return 检测结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public DetectResult detectBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = DetectImageUtils.byteArrayToMat(imageBytes);
			return detector.detect(bgr);
		} finally {
			DetectImageUtils.releaseAll(bgr);
		}
	}

	/**
	 * 检测图像文件。
	 *
	 * @param imagePath 图像文件路径
	 * @return 检测结果
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public DetectResult detectPath(String imagePath) {
		Path p = Paths.get(imagePath);
		if (!Files.exists(p)) {
			throw new MicaAiException(ErrorCode.NOT_FOUND, "文件不存在: " + imagePath);
		}
		try {
			return detectBytes(Files.readAllBytes(p));
		} catch (IOException e) {
			throw new MicaAiException(ErrorCode.UNKNOWN, "读取图像失败: " + imagePath, e);
		}
	}

	/**
	 * 当前配置。
	 *
	 * @return 构造时传入的 {@link DetectConfig}
	 */
	public DetectConfig getConfig() {
		return config;
	}

	/**
	 * 类别名表（配置未指定时返回 COCO 80 类）。
	 *
	 * @return 类别名数组
	 */
	public String[] getLabels() {
		return labels;
	}

	/**
	 * 模型输入边长。
	 *
	 * @return 输入边长；读取失败返回 -1
	 */
	public int modelInputSize() {
		return detector.modelInputSize();
	}

	/**
	 * 模型输出节点个数（NanoDet-Plus 为 6，YOLOX-s 为 1）。
	 *
	 * @return 输出节点个数；读取失败返回 -1
	 */
	public int outputCount() {
		return detector.outputCount();
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
				log.warn("mica-ai-detect: 关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
			}
		}
	}
}
