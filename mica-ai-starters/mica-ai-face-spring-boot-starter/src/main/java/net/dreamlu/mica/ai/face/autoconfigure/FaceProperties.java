/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.autoconfigure;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.nio.file.Path;

/**
 * Face 引擎配置属性。
 *
 * <p>对应 {@code mica.ai.face} 配置前缀。
 * <p>默认模型实现：OpenCV Zoo YuNet + SFace（Apache-2.0，可商用）。
 *
 * @since 1.0.0
 */
@Data
@ConfigurationProperties(prefix = "mica.ai.face")
public class FaceProperties {

	/** 检测 ONNX 模型路径（必填，OpenCV Zoo: face_detection_yunet_*.onnx） */
	private Path detModelPath;

	/** 识别 ONNX 模型路径（必填，OpenCV Zoo: face_recognition_sface_*.onnx） */
	private Path recModelPath;

	/** 模型实现类型，默认 YUNET_SFACE（OpenCV Zoo，Apache-2.0） */
	private FaceConfigModelType modelType = FaceConfigModelType.YUNET_SFACE;

	/** 检测置信度阈值，默认 0.6 */
	private float detScoreThreshold = 0.6f;

	/** NMS IoU 阈值，默认 0.3（YuNet 内部已做 NMS，留作自定义 detector 使用） */
	private float detNmsThreshold = 0.3f;

	/** ONNX 内部线程数 */
	private int intraOpNumThreads = 1;

	/** ONNX 交互线程数 */
	private int interOpNumThreads = 1;

	/**
	 * 与 {@link net.dreamlu.mica.ai.face.config.FaceConfig.ModelType} 对齐的 Spring 配置枚举。
	 *
	 * <p>之所以不复用 core 的枚举，是为了避免 starter 反向依赖 core 的内部细节。
	 */
	public enum FaceConfigModelType {
		/** OpenCV Zoo YuNet + SFace（Apache-2.0，可商用） */
		YUNET_SFACE
	}
}