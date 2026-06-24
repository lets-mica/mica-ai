/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.config;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 人脸 Embedding（识别向量），由 ArcFace 推理得到。
 *
 * <p>buffalo_l / w600k_r50 输出的 embedding 维度为 512，已 L2 归一化。
 *
 * @since 1.0.0
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class FaceEmbedding {

	/** 512 维 L2 归一化向量 */
	private float[] vector;

	/** 来自原图的归属（用于检索时反查） */
	private String userId;

	public FaceEmbedding(float[] vector) {
		this.vector = vector;
	}

	/**
	 * @return 向量维度
	 */
	public int dimension() {
		return vector == null ? 0 : vector.length;
	}
}
