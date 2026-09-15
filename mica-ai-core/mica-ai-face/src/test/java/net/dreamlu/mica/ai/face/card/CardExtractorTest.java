/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.card;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Test;
import org.opencv.core.Point;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * CardExtractor 纯几何单测（不依赖原生库）。
 */
class CardExtractorTest {

	private static final Offset<Double> EPS = Offset.offset(1e-6);

	@Test
	void measuredRatioAxisAlignedRectangle() {
		// 长 200 × 宽 100 的矩形，ratio = 2.0
		float[][] quad = {
			{0f, 0f},
			{200f, 0f},
			{200f, 100f},
			{0f, 100f}
		};
		assertThat(CardExtractor.measuredRatio(quad)).isCloseTo(2.0, EPS);
	}

	@Test
	void measuredRatioIgnoresQuadOrdering() {
		float[][] a = {
			{0f, 0f},
			{200f, 0f},
			{200f, 100f},
			{0f, 100f}
		};
		float[][] b = {
			{200f, 100f},
			{0f, 100f},
			{0f, 0f},
			{200f, 0f}
		};
		assertThat(CardExtractor.measuredRatio(a)).isEqualTo(CardExtractor.measuredRatio(b));
	}

	@Test
	void isConvexAcceptsRectangle() {
		Point[] p = {
			new Point(0, 0),
			new Point(200, 0),
			new Point(200, 100),
			new Point(0, 100)
		};
		assertThat(CardExtractor.isConvex(p)).isTrue();
	}

	@Test
	void isConvexRejectsDegenerateCollinearQuad() {
		Point[] p = {
			new Point(0, 0),
			new Point(50, 0),
			new Point(100, 0),
			new Point(150, 0)
		};
		assertThat(CardExtractor.isConvex(p)).isFalse();
	}
}