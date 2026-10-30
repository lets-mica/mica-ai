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
package net.dreamlu.mica.ai.classify.core;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.classify.config.ClassifyConfig;
import net.dreamlu.mica.ai.classify.detection.PPClassificationDetector;
import net.dreamlu.mica.ai.classify.model.ClassifyResult;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/**
 * PULC 图像分类门面（多模型可插拔）。
 *
 * <p>同一个引擎实例绑定「一个 ONNX 模型 + 一份 label_list」；需要同时用多个 PULC 模型
 * （如文本方向 + 清晰度 + 表格属性）时，各模型分别创建引擎实例即可。
 *
 * <pre>{@code
 * ClassifyConfig config = ClassifyConfig.builder()
 *     .modelPath("model-tools/classify/models/text_image_orientation.onnx")
 *     .labelsPath("model-tools/classify/models/text_image_orientation_labels.txt")
 *     .build();
 * try (ClassifyEngine engine = ClassifyEngine.create(config)) {
 *     ClassifyResult r = engine.classifyBytes(imageBytes);
 *     System.out.println(r.topLabel() + " " + r.topScore());
 * }
 * }</pre>
 *
 * <p>线程安全：内部 detector + ONNX session 线程安全；本类自身仅含构造时冻结的不可变
 * 字段，可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class ClassifyEngine implements AutoCloseable {

	private final PPClassificationDetector detector;
	/**
	 * 获取当前配置。
	 */
	@Getter
	private final ClassifyConfig config;
	private final OrtSession.SessionOptions sessionOptions;

	/**
	 * 构造门面，校验配置并初始化底层分类器。
	 *
	 * @param config 分类配置
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public ClassifyEngine(ClassifyConfig config) {
		Objects.requireNonNull(config, "ClassifyConfig must not be null");
		config.validate();
		this.config = config;
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = OrtSessionFactory.build(config.getOnnx());
		this.detector = new PPClassificationDetector(environment, config, sessionOptions);
		log.info("mica-ai-classify 初始化完成: version={} input={}x{} topK={} resizeShort={}",
			config.getModelVersion(), config.getInputWidth(), config.getInputHeight(),
			config.getTopK(), config.getResizeShort());
	}

	/**
	 * 创建 {@link ClassifyEngine} 实例。
	 *
	 * @param config 分类配置
	 * @return 初始化完成的引擎
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 配置非法或模型加载失败
	 */
	public static ClassifyEngine create(ClassifyConfig config) {
		return new ClassifyEngine(config);
	}

	/**
	 * 对图像做分类。
	 *
	 * @param image 原图 BGR Mat
	 * @return 分类结果；输入为 null / 空图时返回 null
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 推理失败
	 */
	public ClassifyResult classify(Mat image) {
		return detector.classify(image);
	}

	/**
	 * 对图像做分类（字节入参）。
	 *
	 * @param imageBytes 图像字节（png / jpg 等）
	 * @return 分类结果
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public ClassifyResult classifyBytes(byte[] imageBytes) {
		return detector.classifyBytes(imageBytes);
	}

	/**
	 * 对图像做分类（文件路径入参）。
	 *
	 * @param imagePath 图像文件路径
	 * @return 分类结果
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public ClassifyResult classifyPath(String imagePath) {
		return detector.classifyBytes(readFile(imagePath));
	}

	/**
	 * 获取 label_list（行序即类别索引）。
	 *
	 * @return 标签数组（返回副本，避免调用方改写内部状态）
	 */
	public String[] labels() {
		return detector.getLabels().clone();
	}

	/**
	 * 类别数（以 label_list 条目数为准）。
	 *
	 * @return 类别数
	 */
	public int classCount() {
		return detector.classCount();
	}

	/**
	 * 是否多标签模型。
	 *
	 * @return 多标签模式返回 {@code true}
	 */
	public boolean isMultiLabel() {
		return config.isMultiLabel();
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
				log.warn("mica-ai-classify: 关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
			}
		}
	}
}
