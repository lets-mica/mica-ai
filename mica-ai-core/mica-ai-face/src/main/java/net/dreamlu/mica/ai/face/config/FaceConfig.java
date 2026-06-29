/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.config;

import lombok.Builder;
import lombok.Data;

import java.nio.file.Path;

/**
 * Face 引擎配置。
 *
 * <p>使用 Builder 风格构造，必填项（检测/识别模型路径）由 {@link #builder()} 强制提供：
 * <pre>{@code
 * FaceConfig config = FaceConfig.builder()
 *     .detModelPath(Path.of("models/face_detection_yunet_2023mar.onnx"))
 *     .recModelPath(Path.of("models/face_recognition_sface_2021dec.onnx"))
 *     .threshold(0.5f)
 *     .build();
 * }</pre>
 *
 */
@Data
@Builder(toBuilder = true)
public class FaceConfig {

	/**
	 * 人脸检测 ONNX 模型路径（必填，YUNET_SFACE -> face_detection_yunet_*.onnx）
	 */
	private Path detModelPath;

	/**
	 * 人脸识别 ONNX 模型路径（必填，YUNET_SFACE -> face_recognition_sface_*.onnx）
	 */
	private Path recModelPath;

	/**
	 * 检测置信度阈值（0~1），低于此值的人脸被丢弃，默认 0.6。
	 */
	@Builder.Default
	private float detScoreThreshold = 0.6f;

	/**
	 * 检测 NMS IoU 阈值，0~1，默认 0.3。
	 *
	 * <p>注：YuNet 模型内部已做 NMS，此参数预留用于自定义 detector。
	 */
	@Builder.Default
	private float detNmsThreshold = 0.3f;

	/**
	 * ONNX 内部线程数，默认 1
	 */
	@Builder.Default
	private int intraOpNumThreads = 1;

	/**
	 * ONNX 交互线程数，默认 1
	 */
	@Builder.Default
	private int interOpNumThreads = 1;

	/**
	 * 检测输入边长，YuNet 推荐 320；自定义 detector 可调整。
	 */
	@Builder.Default
	private int detInputSize = 320;

	/**
	 * 识别输入边长，SFace / ArcFace 固定 112。
	 */
	@Builder.Default
	private int recInputSize = 112;

}
