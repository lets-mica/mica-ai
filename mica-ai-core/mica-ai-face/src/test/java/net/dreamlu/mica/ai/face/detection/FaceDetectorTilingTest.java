/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.detection;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * FaceDetector 滑窗分块起点算法的纯几何单测。
 *
 * <p>仅依赖坐标算法，不依赖 ONNX / OpenCV 原生库。
 */
class FaceDetectorTilingTest {

	@Test
	void singleBlockReturnsZero() {
		int[] starts = FaceDetector.tileStarts(300, 640, 480);
		assertThat(starts).containsExactly(0);
	}

	@Test
	void coversLengthWithOverlap() {
		// length = 1000, tile = 480, overlap = 0.3 → step = 336
		int[] starts = FaceDetector.tileStarts(1000, 480, 336);
		// 期望起点：0, 336, 672 + 右贴边补齐 520
		assertThat(starts[0]).isEqualTo(0);
		assertThat(starts[starts.length - 1]).isEqualTo(1000 - 480);
		// 任意相邻起点之差应 <= step
		for (int i = 1; i < starts.length; i++) {
			assertThat(starts[i] - starts[i - 1]).isLessThanOrEqualTo(336);
		}
	}
}