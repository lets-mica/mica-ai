/*
 * Copyright (c) 2019-2029, Dreamlu 卢春梦 (596392912@qq.com & dreamlu.net).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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