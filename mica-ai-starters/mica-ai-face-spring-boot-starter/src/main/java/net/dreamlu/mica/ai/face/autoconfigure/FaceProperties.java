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
 *
 * @since 1.0.0
 */
@Data
@ConfigurationProperties(prefix = "mica.ai.face")
public class FaceProperties {

	/** 检测 ONNX 模型路径（必填，buffalo_l -> det_10g.onnx） */
	private Path detModelPath;

	/** 识别 ONNX 模型路径（必填，buffalo_l -> w600k_r50.onnx） */
	private Path recModelPath;

	/** 检测置信度阈值 */
	private float detScoreThreshold = 0.5f;

	/** NMS IoU 阈值 */
	private float detNmsThreshold = 0.4f;

	/** ONNX 内部线程数 */
	private int intraOpNumThreads = 1;

	/** ONNX 交互线程数 */
	private int interOpNumThreads = 1;
}
