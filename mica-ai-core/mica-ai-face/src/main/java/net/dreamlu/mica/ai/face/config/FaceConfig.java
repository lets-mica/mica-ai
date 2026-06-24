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
 *     .detModelPath(Path.of("models/det_10g.onnx"))
 *     .recModelPath(Path.of("models/w600k_r50.onnx"))
 *     .threshold(0.45f)        // 1:1 比对阈值
 *     .build();
 * }</pre>
 */
@Data
@Builder
public class FaceConfig {

	/** 人脸检测 ONNX 模型路径（必填，buffalo_l -> det_10g.onnx） */
	private Path detModelPath;

	/** 人脸识别 ONNX 模型路径（必填，buffalo_l -> w600k_r50.onnx） */
	private Path recModelPath;

	/** 检测置信度阈值（0~1），低于此值的人脸被丢弃，默认 0.5 */
	@Builder.Default
	private float detScoreThreshold = 0.5f;

	/** 检测 NMS IoU 阈值，0~1，默认 0.4 */
	@Builder.Default
	private float detNmsThreshold = 0.4f;

	/** ONNX 内部线程数，默认 1 */
	@Builder.Default
	private int intraOpNumThreads = 1;

	/** ONNX 交互线程数，默认 1 */
	@Builder.Default
	private int interOpNumThreads = 1;

	/** 检测输入边长，RetinaFace 固定 640 */
	@Builder.Default
	private int detInputSize = 640;

	/** 识别输入边长，ArcFace w600k_r50 固定 112 */
	@Builder.Default
	private int recInputSize = 112;
}
