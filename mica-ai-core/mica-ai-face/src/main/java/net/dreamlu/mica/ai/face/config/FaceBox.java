/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.config;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 人脸检测框 + 5 关键点。
 *
 * <p>坐标系：原图坐标，y 向下。
 *
 * @since 1.0.0
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class FaceBox {

	/** 边界框左上角 x */
	private float x1;
	/** 边界框左上角 y */
	private float y1;
	/** 边界框右下角 x */
	private float x2;
	/** 边界框右下角 y */
	private float y2;
	/** 检测置信度，0~1 */
	private float score;
	/** 5 个关键点（眼睛、鼻尖、嘴角）flatten 后 = 长度 10 的数组 */
	private float[] landmarks;

	/**
	 * @return 边界框宽度
	 */
	public float width() {
		return x2 - x1;
	}

	/**
	 * @return 边界框高度
	 */
	public float height() {
		return y2 - y1;
	}

	/**
	 * @return 边界框面积
	 */
	public float area() {
		return width() * height();
	}
}
