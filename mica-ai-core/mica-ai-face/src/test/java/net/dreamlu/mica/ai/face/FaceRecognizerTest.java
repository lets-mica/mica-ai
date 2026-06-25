/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.engine.FaceRecognizer;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link FaceRecognizer} 接口静态向量工具测试（L2 normalize / cosine similarity）。
 *
 * <p>不依赖 ONNX Runtime，纯算法验证。
 */
class FaceRecognizerTest {

	private static final float EPS = 1e-6f;

	@Test
	void l2Normalize_returns_unit_vector() {
		float[] v = {3f, 4f};  // 长度 5
		float[] out = FaceRecognizer.l2Normalize(v);
		double len = Math.sqrt(out[0] * out[0] + out[1] * out[1]);
		assertEquals(1.0, len, EPS);
		assertEquals(0.6f, out[0], EPS);
		assertEquals(0.8f, out[1], EPS);
	}

	@Test
	void l2Normalize_zero_vector_returns_input_unchanged() {
		float[] v = {0f, 0f, 0f};
		float[] out = FaceRecognizer.l2Normalize(v);
		assertSame(v, out);
	}

	@Test
	void cosineSimilarity_identical_is_one() {
		float[] a = {1f, 0f, 0f};
		assertEquals(1f, FaceRecognizer.cosineSimilarity(a, a), EPS);
	}

	@Test
	void cosineSimilarity_orthogonal_is_zero() {
		float[] a = {1f, 0f, 0f};
		float[] b = {0f, 1f, 0f};
		assertEquals(0f, FaceRecognizer.cosineSimilarity(a, b), EPS);
	}

	@Test
	void cosineSimilarity_opposite_is_minus_one() {
		float[] a = {1f, 0f};
		float[] b = {-1f, 0f};
		assertEquals(-1f, FaceRecognizer.cosineSimilarity(a, b), EPS);
	}

	@Test
	void cosineSimilarity_length_mismatch_throws() {
		float[] a = {1f, 0f};
		float[] b = {1f, 0f, 0f};
		MicaAiException ex = assertThrows(MicaAiException.class,
			() -> FaceRecognizer.cosineSimilarity(a, b));
		assertTrue(ex.getMessage().contains("length"), () -> "message: " + ex.getMessage());
	}

	@Test
	void cosineSimilarity_same_person_high_score() {
		// 模拟 SFace 的"同人" embedding（同方向稍加噪声）
		float[] base = new float[512];
		for (int i = 0; i < 512; i++) {
			base[i] = (float) Math.sin(i * 0.1);
		}
		float[] baseNorm = FaceRecognizer.l2Normalize(base);
		float[] noisy = baseNorm.clone();
		noisy[0] += 0.05f;
		noisy[1] -= 0.05f;
		float[] noisyNorm = FaceRecognizer.l2Normalize(noisy);

		float score = FaceRecognizer.cosineSimilarity(baseNorm, noisyNorm);
		// 加了少量噪声后相似度应仍然很高（> 0.95）
		assertTrue(score > 0.95f, () -> "score was " + score);
	}
}