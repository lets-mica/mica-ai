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
package net.dreamlu.mica.ai.track.model;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link BBox} 基础计算单测。
 */
class BBoxTest {

	@Test
	void emptyBoxReturnsTrue() {
		assertThat(new BBox(10f, 10f, 10f, 10f, 0.9f).isEmpty()).isTrue();
		assertThat(new BBox(10f, 10f, 5f, 15f, 0.9f).isEmpty()).isTrue(); // x2 < x1
	}

	@Test
	void nonEmptyBoxReturnsFalse() {
		assertThat(new BBox(10f, 10f, 20f, 20f, 0.9f).isEmpty()).isFalse();
	}

	@Test
	void cxcyWidthHeight() {
		BBox b = new BBox(0f, 0f, 10f, 20f, 0.9f);
		assertThat(b.cx()).isEqualTo(5f);
		assertThat(b.cy()).isEqualTo(10f);
		assertThat(b.width()).isEqualTo(10f);
		assertThat(b.height()).isEqualTo(20f);
	}

	@Test
	void iouSelfIsOne() {
		BBox b = new BBox(0f, 0f, 10f, 10f, 0.9f);
		assertThat(b.iou(b)).isEqualTo(1f);
	}

	@Test
	void iouDisjointIsZero() {
		BBox a = new BBox(0f, 0f, 10f, 10f, 0.9f);
		BBox b = new BBox(20f, 20f, 30f, 30f, 0.9f);
		assertThat(a.iou(b)).isEqualTo(0f);
		assertThat(b.iou(a)).isEqualTo(0f);
	}

	@Test
	void iouPartialOverlap() {
		// 10x10 方框，相交 5x10，总面积 100+100-50=150
		BBox a = new BBox(0f, 0f, 10f, 10f, 0.9f);
		BBox b = new BBox(5f, 0f, 15f, 10f, 0.9f);
		assertThat(a.iou(b)).isEqualTo(50f / 150f);
	}

	@Test
	void iouEmptyBoxIsZero() {
		BBox a = new BBox(0f, 0f, 10f, 10f, 0.9f);
		BBox empty = new BBox(0f, 0f, 0f, 0f, 0.9f);
		assertThat(a.iou(empty)).isEqualTo(0f);
		assertThat(empty.iou(a)).isEqualTo(0f);
	}

	@Test
	void classIdDefaultsToZero() {
		BBox b = new BBox(0f, 0f, 10f, 10f, 0.9f);
		assertThat(b.getClassId()).isEqualTo(0);
	}
}
