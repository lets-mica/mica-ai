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
package net.dreamlu.mica.ai.face.model;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.face.config.ModelConfig;

import java.util.Objects;

/**
 * mica-ai-face 模型管理器。
 *
 * <p>持有共享的 {@link OrtEnvironment} 与 ONNX 会话基础配置（{@link OrtSessionOptions}），
 * 并为每个模型创建独立的 {@link OnnxModelSession}：检测、特征、活体三者彼此解耦。
 */
@Slf4j
@Getter
public class ModelManager implements AutoCloseable {

	private final OrtEnvironment environment;
	private final OrtSessionOptions onnxOptions;
	private final OrtSession.SessionOptions sessionOptions;

	private final OnnxModelSession detection;
	private final OnnxModelSession recognition;
	private final OnnxModelSession liveness;

	private final ModelConfig config;

	/**
	 * 按配置加载检测、识别、（可选的）活体 ONNX 模型。
	 *
	 * @param config 模型配置，不能为 {@code null}
	 * @throws NullPointerException config 为 {@code null} 时抛出
	 */
	public ModelManager(ModelConfig config) {
		this.config = Objects.requireNonNull(config, "ModelConfig must not be null");
		this.environment = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR);
		this.onnxOptions = config.getOnnx() != null ? config.getOnnx() : OrtSessionOptions.defaults();
		this.sessionOptions = OrtSessionFactory.build(this.onnxOptions);
		this.detection = new OnnxModelSession(environment, config.getDetectionModelPath(), sessionOptions, "detection");
		this.recognition = new OnnxModelSession(environment, config.getRecognitionModelPath(), sessionOptions, "recognition");
		if (config.getLivenessModelPath() != null && !config.getLivenessModelPath().isEmpty()) {
			this.liveness = new OnnxModelSession(environment, config.getLivenessModelPath(), sessionOptions, "liveness");
		} else {
			this.liveness = null;
		}
	}

	/**
	 * 按配置创建模型管理器。
	 *
	 * @param config 模型配置
	 * @return 模型管理器实例
	 */
	public static ModelManager create(ModelConfig config) {
		return new ModelManager(config);
	}

	/**
	 * 获取人脸检测模型会话。
	 *
	 * @return 检测模型 OrtSession
	 */
	public OrtSession getDetectionSession() {
		return detection.getSession();
	}

	/**
	 * 获取人脸识别模型会话。
	 *
	 * @return 识别模型 OrtSession
	 */
	public OrtSession getRecognitionSession() {
		return recognition.getSession();
	}

	/**
	 * 获取活体检测模型会话，未配置活体模型时返回 {@code null}。
	 *
	 * @return 活体模型 OrtSession，未加载时为 {@code null}
	 */
	public OrtSession getLivenessSession() {
		return liveness != null ? liveness.getSession() : null;
	}

	@Override
	public void close() throws Exception {
		if (detection != null) {
			detection.close();
		}
		if (recognition != null) {
			recognition.close();
		}
		if (liveness != null) {
			liveness.close();
		}
		try {
			sessionOptions.close();
		} catch (Exception e) {
			log.warn("关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
		}
	}
}
