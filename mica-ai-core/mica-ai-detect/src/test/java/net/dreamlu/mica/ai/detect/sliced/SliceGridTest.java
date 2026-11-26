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
package net.dreamlu.mica.ai.detect.sliced;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link SliceGrid} 网格计算测试（纯几何，不依赖 OpenCV / ONNX）。
 */
class SliceGridTest {

	/**
	 * 取某一维上的切片起点（去重后升序）。
	 */
	private static List<Integer> startsOf(List<Slice> slices, boolean horizontal) {
		List<Integer> starts = new ArrayList<>();
		for (Slice slice : slices) {
			int start = horizontal ? slice.getX() : slice.getY();
			if (!starts.contains(start)) {
				starts.add(start);
			}
		}
		starts.sort(Integer::compareTo);
		return starts;
	}

	@Test
	void computesGridForLargeImage() {
		SliceGrid grid = SliceGrid.of(1200, 1500, 416, 312);

		// 宽 1200 → 起点 0 / 312 / 624 / 784 共 4 列；高 1500 → 起点 0 / 312 / 624 / 936 / 1084 共 5 行
		assertThat(grid.getSliceCount()).isEqualTo(20);
		assertThat(grid.getWidth()).isEqualTo(1200);
		assertThat(grid.getHeight()).isEqualTo(1500);
		assertThat(grid.getSize()).isEqualTo(416);
		assertThat(grid.getStep()).isEqualTo(312);
	}

	@Test
	void lastColumnFallsBackToCoverRightEdge() {
		SliceGrid grid = SliceGrid.of(1200, 1500, 416, 312);

		List<Integer> starts = startsOf(grid.getSlices(), true);
		assertThat(starts).containsExactly(0, 312, 624, 784);
		// 末列回退到「边界 - 边长」，保证右侧不被漏掉
		assertThat(starts.get(starts.size() - 1) + 416).isEqualTo(1200);
	}

	@Test
	void coversWholeImageWithoutGap() {
		SliceGrid grid = SliceGrid.of(1234, 987, 416, 312);

		List<Integer> xStarts = startsOf(grid.getSlices(), true);
		List<Integer> yStarts = startsOf(grid.getSlices(), false);

		assertThat(xStarts.get(0)).isLessThanOrEqualTo(0);
		assertThat(yStarts.get(0)).isLessThanOrEqualTo(0);
		for (int i = 1; i < xStarts.size(); i++) {
			// 相邻切片的间隔必须小于边长，否则中间会漏掉一条未被任何切片覆盖的区域
			assertThat(xStarts.get(i) - xStarts.get(i - 1)).isLessThanOrEqualTo(416);
		}
		for (int i = 1; i < yStarts.size(); i++) {
			assertThat(yStarts.get(i) - yStarts.get(i - 1)).isLessThanOrEqualTo(416);
		}
		assertThat(xStarts.get(xStarts.size() - 1) + 416).isGreaterThanOrEqualTo(1234);
		assertThat(yStarts.get(yStarts.size() - 1) + 416).isGreaterThanOrEqualTo(987);
	}

	@Test
	void usesSingleSliceWhenImageSmallerThanTile() {
		SliceGrid grid = SliceGrid.of(300, 200, 416, 312);

		assertThat(grid.getSliceCount()).isEqualTo(1);
		assertThat(grid.getSlices().get(0).getX()).isZero();
		assertThat(grid.getSlices().get(0).getY()).isZero();
	}

	@Test
	void usesSingleSliceWhenImageEqualsTile() {
		SliceGrid grid = SliceGrid.of(416, 416, 416, 312);

		assertThat(grid.getSliceCount()).isEqualTo(1);
	}

	@Test
	void sliceExposesOriginalCoordinateEdges() {
		Slice slice = new Slice(312, 208, 416);

		assertThat(slice.getX()).isEqualTo(312);
		assertThat(slice.getY()).isEqualTo(208);
		assertThat(slice.getSize()).isEqualTo(416);
		assertThat(slice.getX2()).isEqualTo(728);
		assertThat(slice.getY2()).isEqualTo(624);
	}

	@Test
	void sliceOverlapAreaIsZeroWhenDisjoint() {
		Slice slice = new Slice(0, 0, 416);

		assertThat(slice.overlapArea(500, 500, 600, 600)).isZero();
		assertThat(slice.overlapArea(100, 100, 200, 200)).isEqualTo(100L * 100L);
	}

	@Test
	void rejectsIllegalArguments() {
		assertThatThrownBy(() -> SliceGrid.of(0, 100, 416, 312))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("width");
		assertThatThrownBy(() -> SliceGrid.of(100, 100, 0, 312))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("size");
		assertThatThrownBy(() -> SliceGrid.of(100, 100, 416, 0))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("step");
	}

}
