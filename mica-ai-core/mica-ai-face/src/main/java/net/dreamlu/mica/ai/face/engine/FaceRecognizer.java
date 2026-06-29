/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.engine;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.config.FaceEmbedding;

import java.awt.image.BufferedImage;
import java.io.Closeable;

/**
 * 人脸识别器抽象。
 *
 * <p>实现该接口即可接入 mica-ai-face 的人脸识别主流程。
 * 输入应当是已经对齐到标准尺寸（如 112x112）的人脸图像，
 * 具体对齐逻辑由 {@link net.dreamlu.mica.ai.face.utils.ImageUtils#align(BufferedImage, net.dreamlu.mica.ai.face.config.FaceBox, int)} 提供。
 *
 * <p>内置默认实现 {@link SFaceRecognizer}（OpenCV Zoo，Apache-2.0，可商用）。
 *
 * @author dreamlu
 * @since 1.0.0
 */
public interface FaceRecognizer extends Closeable {

	/**
	 * 从一张已经对齐到标准尺寸的人脸图片中提取 Embedding 向量。
	 *
	 * <p>返回值 {@link FaceEmbedding#getVector()} 为 L2 归一化后的向量，
	 * 任意两个向量的点积 = 余弦相似度。
	 *
	 * @param alignedFace 对齐后的人脸图像（建议 BGR / 112x112）
	 * @return 人脸 Embedding
	 */
	FaceEmbedding extract(BufferedImage alignedFace);

	// --------------------------------------------------------------------
	// 通用向量工具（与具体模型无关，作为静态工具方法暴露）
	// --------------------------------------------------------------------

	/**
	 * L2 归一化一个 float 向量。
	 *
	 * <p>如果输入是零向量，原样返回。
	 */
	static float[] l2Normalize(float[] v) {
		double sum = 0;
		for (float f : v) {
			sum += (double) f * f;
		}
		float norm = (float) Math.sqrt(sum);
		if (norm == 0f) {
			return v;
		}
		float[] out = new float[v.length];
		for (int i = 0; i < v.length; i++) {
			out[i] = v[i] / norm;
		}
		return out;
	}

	/**
	 * 计算两个同维向量的余弦相似度（点积）。
	 *
	 * <p>要求两个向量都已 L2 归一化，否则语义不是"余弦"。
	 */
	static float cosineSimilarity(float[] a, float[] b) {
		if (a.length != b.length) {
			throw new MicaAiException("Embedding length mismatch: " + a.length + " vs " + b.length);
		}
		float sum = 0;
		for (int i = 0; i < a.length; i++) {
			sum += a[i] * b[i];
		}
		return sum;
	}
}
