/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 人脸检测框。
 *
 * <p>描述人脸检测模型（YuNet）输出的单个人脸区域，包含边界框坐标、置信度与 5 个关键点
 * (左眼、右眼、鼻尖、左嘴角、右嘴角)。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class FaceBox {

	private float x1;
	private float y1;
	private float x2;
	private float y2;
	private float score;
	private float[][] landmarks;
}