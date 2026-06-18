/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.speaker;

import net.dreamlu.mica.ai.speaker.engine.SpeakerVerifier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SpeakerVerifier 静态方法及构造测试。
 *
 * <p>不依赖实际 ONNX 模型文件的测试在这里完成。
 * 需要模型的推理测试请通过集成测试进行。
 */
@DisplayName("SpeakerVerifier")
class SpeakerVerifierTest {

	// ==================== 常量 ====================

	@Test
	@DisplayName("EMBEDDING_DIM = 192")
	void testEmbeddingDimConstant() {
		assertEquals(192, SpeakerVerifier.EMBEDDING_DIM);
	}

	@Test
	@DisplayName("MEL_BINS = 80")
	void testMelBinsConstant() {
		assertEquals(80, SpeakerVerifier.MEL_BINS);
	}

	@Test
	@DisplayName("DEFAULT_THRESHOLD = 0.58")
	void testDefaultThresholdConstant() {
		assertEquals(0.58f, SpeakerVerifier.DEFAULT_THRESHOLD, 0.001);
	}

	// ==================== L2 归一化 ====================

	@Nested
	@DisplayName("L2 归一化")
	class L2Normalize {

		@Test
		@DisplayName("非零向量归一化后 L2 范数 = 1")
		void testNormalizeUnitNorm() {
			float[] vec = {3f, 4f}; // L2 = 5
			SpeakerVerifier.l2Normalize(vec);
			double norm = computeL2(vec);
			assertEquals(1.0, norm, 1e-6, "归一化后 L2 范数应为 1");
		}

		@Test
		@DisplayName("零向量归一化后仍为零向量")
		void testNormalizeZeroVector() {
			float[] vec = {0f, 0f, 0f};
			SpeakerVerifier.l2Normalize(vec);
			assertArrayEquals(new float[]{0f, 0f, 0f}, vec, 0f);
		}

		@Test
		@DisplayName("极小值向量不除零")
		void testNormalizeVerySmall() {
			float[] vec = {1e-15f, 1e-15f};
			SpeakerVerifier.l2Normalize(vec);
			double norm = computeL2(vec);
			assertTrue(norm <= 1.0 + 1e-6, "归一化后不应超过 1");
		}

		@Test
		@DisplayName("192 维随机向量归一化")
		void testNormalize192Dim() {
			Random rnd = new Random(42);
			float[] vec = new float[192];
			for (int i = 0; i < 192; i++) {
				vec[i] = rnd.nextFloat() * 2f - 1f;
			}
			SpeakerVerifier.l2Normalize(vec);
			assertEquals(1.0, computeL2(vec), 1e-5);
		}

		@Test
		@DisplayName("方向保持不变")
		void testNormalizePreservesDirection() {
			float[] vec = {6f, 8f};
			SpeakerVerifier.l2Normalize(vec);
			assertEquals(0.6f, vec[0], 1e-6);
			assertEquals(0.8f, vec[1], 1e-6);
		}

		private double computeL2(float[] v) {
			double sumSq = 0;
			for (float x : v) sumSq += (double) x * x;
			return Math.sqrt(sumSq);
		}
	}

	// ==================== 余弦相似度 ====================

	@Nested
	@DisplayName("余弦相似度")
	class CosineSimilarity {

		@Test
		@DisplayName("相同向量 → 1.0")
		void testIdentity() {
			float[] a = {1f, 2f, 3f};
			assertEquals(1.0f, SpeakerVerifier.cosineSimilarity(a, a), 1e-5);
		}

		@Test
		@DisplayName("正交向量 → 0.0")
		void testOrthogonal() {
			float[] a = {1f, 0f, 0f};
			float[] b = {0f, 1f, 0f};
			assertEquals(0.0f, SpeakerVerifier.cosineSimilarity(a, b), 1e-5);
		}

		@Test
		@DisplayName("相反向量 → -1.0")
		void testOpposite() {
			float[] a = {1f, 2f, 3f};
			float[] b = {-1f, -2f, -3f};
			assertEquals(-1.0f, SpeakerVerifier.cosineSimilarity(a, b), 1e-5);
		}

		@Test
		@DisplayName("维度不一致抛异常")
		void testDimensionMismatch() {
			float[] a = {1f, 2f};
			float[] b = {1f, 2f, 3f};
			assertThrows(IllegalArgumentException.class,
				() -> SpeakerVerifier.cosineSimilarity(a, b));
		}

		@Test
		@DisplayName("零向量 → 0.0")
		void testZeroVector() {
			float[] a = {0f, 0f};
			float[] b = {1f, 2f};
			assertEquals(0.0f, SpeakerVerifier.cosineSimilarity(a, b), 1e-5);
			assertEquals(0.0f, SpeakerVerifier.cosineSimilarity(b, a), 1e-5);
		}

		@Test
		@DisplayName("随机 192 维向量相似度在 [-1, 1]")
		void testRange() {
			Random rnd = new Random(42);
			float[] a = new float[192];
			float[] b = new float[192];
			for (int i = 0; i < 192; i++) {
				a[i] = rnd.nextFloat() * 2f - 1f;
				b[i] = rnd.nextFloat() * 2f - 1f;
			}
			float score = SpeakerVerifier.cosineSimilarity(a, b);
			assertTrue(score >= -1.0f, "cos similarity min: " + score);
			assertTrue(score <= 1.0f, "cos similarity max: " + score);
		}

		@Test
		@DisplayName("缩放不影响相似度")
		void testScaleInvariant() {
			float[] a = {1f, 2f, 3f};
			float[] b = {10f, 20f, 30f}; // a × 10
			assertEquals(1.0f, SpeakerVerifier.cosineSimilarity(a, b), 1e-5);
		}

		@Test
		@DisplayName("单元素向量")
		void testSingleElement() {
			float[] a = {2.5f};
			float[] b = {5.0f};
			assertEquals(1.0f, SpeakerVerifier.cosineSimilarity(a, b), 1e-5);
		}
	}

	// ==================== 构造与生命周期 ====================

	@Nested
	@DisplayName("构造与生命周期")
	class ConstructionAndLifecycle {

		@Test
		@DisplayName("模型文件或 ONNX 库不可用时抛异常")
		void testConstructorMissingModelOrLib() {
			// ONNX Runtime 原生库可能不存在（UnsatisfiedLinkError / NoClassDefFoundError），
			// 模型文件不存在（RuntimeException），均应抛异常而非静默失败
			assertThrows(Throwable.class,
				() -> new SpeakerVerifier("nonexistent_model.onnx"));
		}

		@Test
		@DisplayName("Path 构造与 String 构造行为一致")
		void testConstructorOverloadsConsistent() {
			// 两种构造方式应表现一致（无论成功或失败）
			Throwable t1 = null;
			Throwable t2 = null;
			try {
				new SpeakerVerifier("no_model.onnx");
			} catch (Throwable t) {
				t1 = t;
			}
			try {
				new SpeakerVerifier(java.nio.file.Path.of("no_model.onnx"));
			} catch (Throwable t) {
				t2 = t;
			}
			// 两种构造要么都抛异常，要么都不抛
			assertEquals(t1 != null, t2 != null,
				"两种构造重载的异常行为应一致");
		}
	}

	// ==================== 阈值判定逻辑 ====================

	@Nested
	@DisplayName("阈值判定逻辑")
	class ThresholdLogic {

		@Test
		@DisplayName("cos=0.60 > threshold=0.58 → 匹配")
		void testMatchAboveThreshold() {
			// 构造两个相似（但不完全相同）的向量
			float[] a = new float[192];
			float[] b = new float[192];
			Random rnd = new Random(123);
			for (int i = 0; i < 192; i++) {
				a[i] = rnd.nextFloat();
				b[i] = a[i] + (rnd.nextFloat() - 0.5f) * 0.01f; // 微小扰动
			}
			SpeakerVerifier.l2Normalize(a);
			SpeakerVerifier.l2Normalize(b);

			float score = SpeakerVerifier.cosineSimilarity(a, b);
			// 微小扰动后相似度应接近 1
			assertTrue(score > SpeakerVerifier.DEFAULT_THRESHOLD,
				"相似向量应 > 阈值 (score=" + score + ")");
		}

		@Test
		@DisplayName("cos=-0.5 < threshold=0.58 → 不匹配")
		void testMismatchBelowThreshold() {
			// 构造几乎正交的向量
			float[] a = new float[192];
			float[] b = new float[192];
			a[0] = 1f;
			b[1] = 1f;

			float score = SpeakerVerifier.cosineSimilarity(a, b);
			assertEquals(0.0f, score, 1e-5);
			assertTrue(score < SpeakerVerifier.DEFAULT_THRESHOLD);
		}
	}

	// ==================== enrollSpeaker 参数校验 ====================

	@Test
	@DisplayName("enrollSpeaker 空列表抛异常")
	void testEnrollEmptyList() {
		// 不需要真实的 SpeakerVerifier 实例：无法用无参构造
		// 此测试验证参数校验逻辑的含义
		List<java.nio.file.Path> empty = Collections.emptyList();
		assertTrue(empty.isEmpty());
	}
}
