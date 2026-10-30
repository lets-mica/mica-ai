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
package net.dreamlu.mica.ai.detect.postprocess;

import net.dreamlu.mica.ai.detect.model.DetectBox;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link Nms} 单元测试。
 */
class NmsTest {

	private static DetectBox box(float x1, float y1, float x2, float y2, float score, int classId) {
		return new DetectBox(x1, y1, x2, y2, score, classId);
	}

	@Test
	void emptyInputReturnsEmptyList() {
		assertThat(Nms.suppress(Collections.<DetectBox>emptyList(), 0.5f, true, 10)).isEmpty();
		assertThat(Nms.suppress(null, 0.5f, true, 10)).isEmpty();
	}

	@Test
	void keepsHighestScoreWhenBoxesOverlap() {
		DetectBox high = box(0, 0, 100, 100, 0.9f, 0);
		DetectBox low = box(5, 5, 100, 100, 0.5f, 0);

		List<DetectBox> kept = Nms.suppress(Arrays.asList(low, high), 0.5f, true, 100);

		assertThat(kept).hasSize(1);
		assertThat(kept.get(0).getScore()).isEqualTo(0.9f);
	}

	@Test
	void keepsNonOverlappingBoxes() {
		List<DetectBox> kept = Nms.suppress(Arrays.asList(
			box(0, 0, 50, 50, 0.9f, 0),
			box(200, 200, 250, 250, 0.8f, 0)), 0.5f, true, 100);

		assertThat(kept).hasSize(2);
	}

	@Test
	void classAwareKeepsDifferentClasses() {
		List<DetectBox> kept = Nms.suppress(Arrays.asList(
			box(0, 0, 100, 100, 0.9f, 0),
			box(2, 2, 100, 100, 0.8f, 1)), 0.5f, false, 100);

		assertThat(kept).hasSize(2);
		assertThat(kept).extracting(DetectBox::getClassId).containsExactlyInAnyOrder(0, 1);
	}

	@Test
	void classAgnosticSuppressesDifferentClasses() {
		List<DetectBox> kept = Nms.suppress(Arrays.asList(
			box(0, 0, 100, 100, 0.9f, 0),
			box(2, 2, 100, 100, 0.8f, 1)), 0.5f, true, 100);

		assertThat(kept).hasSize(1);
		assertThat(kept.get(0).getClassId()).isEqualTo(0);
	}

	@Test
	void respectsMaxDetections() {
		List<DetectBox> boxes = new ArrayList<>();
		for (int i = 0; i < 10; i++) {
			boxes.add(box(i * 60, 0, i * 60 + 50, 50, 0.9f - i * 0.01f, 0));
		}
		List<DetectBox> kept = Nms.suppress(boxes, 0.5f, true, 3);

		assertThat(kept).hasSize(3);
		assertThat(kept.get(0).getScore()).isGreaterThan(kept.get(1).getScore());
	}

	@Test
	void boundaryIouEqualToThresholdIsKept() {
		// IoU = 5000 / 10000 = 0.5，阈值同为 0.5 → 严格大于才抑制，故两条都保留
		List<DetectBox> kept = Nms.suppress(Arrays.asList(
			box(0, 0, 100, 100, 0.9f, 0),
			box(0, 0, 50, 100, 0.8f, 0)), 0.5f, true, 100);

		assertThat(kept).hasSize(2);
	}

	@Test
	void resultIsSortedByScoreDescending() {
		List<DetectBox> kept = Nms.suppress(Arrays.asList(
			box(0, 0, 10, 10, 0.3f, 0),
			box(100, 0, 110, 10, 0.9f, 0),
			box(200, 0, 210, 10, 0.6f, 0)), 0.5f, true, 100);

		assertThat(kept).extracting(DetectBox::getScore).containsExactly(0.9f, 0.6f, 0.3f);
	}
}
