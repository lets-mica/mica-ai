/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.recognition;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * FeatureExtractor L2 归一化单测。
 *
 * <p>不依赖 ONNX 原生库：只测 {@link FeatureExtractor#l2NormalizeCopy(float[])}
 * 与 {@link FeatureExtractor#compare(float[], float[])} 的纯算法。
 */
class FeatureExtractorL2Test {

	private static final Offset<Double> EPS = Offset.offset(1e-6);
	private static final Offset<Float> EPS_F = Offset.offset(1e-6f);

	@Test
	void l2NormalizeProducesUnitVector() {
		float[] v = {3f, 4f};
		float[] out = FeatureExtractor.l2NormalizeCopy(v);
		double sum = 0;
		for (float f : out) {
			sum += f * f;
		}
		assertThat(Math.sqrt(sum)).isCloseTo(1.0, EPS);
	}

	@Test
	void compareReturnsDotProductForUnitVectors() {
		float[] a = FeatureExtractor.l2NormalizeCopy(new float[]{1f, 0f, 0f});
		float[] b = FeatureExtractor.l2NormalizeCopy(new float[]{0f, 1f, 0f});
		// 正交向量余弦相似度 = 0
		assertThat(FeatureExtractor.compare(a, b)).isCloseTo(0f, EPS_F);
	}
}