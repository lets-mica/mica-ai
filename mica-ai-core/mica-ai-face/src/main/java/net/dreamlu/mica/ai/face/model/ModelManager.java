/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.model;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.face.config.ModelConfig;
import net.dreamlu.mica.ai.face.onnx.OrtSessionFactory;
import net.dreamlu.mica.ai.face.onnx.OrtSessionOptions;

import javax.annotation.PreDestroy;

import java.util.Objects;

/**
 * mica-ai-face 模型管理器。
 *
 * <p>持有共享的 {@link OrtEnvironment} 与 ONNX 会话基础配置（{@link OrtSessionOptions}），
 * 并为每个模型创建独立的 {@link OnnxModelSession}：检测、特征、活体三者彼此解耦。
 */
@Slf4j
@Getter
public class ModelManager {

	private final OrtEnvironment environment;
	private final OrtSessionOptions onnxOptions;
	private final OrtSession.SessionOptions sessionOptions;

	private final OnnxModelSession detection;
	private final OnnxModelSession recognition;
	private final OnnxModelSession liveness;

	private final ModelConfig config;

	public ModelManager(ModelConfig config) {
		this.config = Objects.requireNonNull(config, "ModelConfig must not be null");
		this.environment = OrtEnvironment.getEnvironment();
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

	public static ModelManager create(ModelConfig config) {
		return new ModelManager(config);
	}

	public OrtSession getDetectionSession() {
		return detection.getSession();
	}

	public OrtSession getRecognitionSession() {
		return recognition.getSession();
	}

	public OrtSession getLivenessSession() {
		return liveness != null ? liveness.getSession() : null;
	}

	@PreDestroy
	public void destroy() {
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
		} catch (Throwable e) {
			log.warn("关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
		}
	}
}