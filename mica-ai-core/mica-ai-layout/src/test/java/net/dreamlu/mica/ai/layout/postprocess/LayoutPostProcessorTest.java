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
package net.dreamlu.mica.ai.layout.postprocess;

import net.dreamlu.mica.ai.layout.config.LayoutConfig;
import net.dreamlu.mica.ai.layout.model.LayoutLabel;
import net.dreamlu.mica.ai.layout.model.LayoutResult;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class LayoutPostProcessorTest {

	private static float[] box(LayoutLabel label, float score, float x1, float y1,
							   float x2, float y2, float order) {
		return new float[]{label.getIndex(), score, x1, y1, x2, y2, order};
	}

	@Test
	void shouldReverseLetterboxToOriginalCoords() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.9f, 100f, 50f, 300f, 250f, 1f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 0.5d, 0, 0, 1000, 500);
		assertThat(out).hasSize(1);
		assertThat(out.get(0).getBoundingBox()).containsExactly(200, 100, 600, 500);
	}

	@Test
	void shouldSubtractPaddingBeforeScaling() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.9f, 150f, 70f, 350f, 270f, 1f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 0.5d, 100, 20, 1000, 500);
		assertThat(out.get(0).getBoundingBox()).containsExactly(100, 100, 500, 500);
	}

	@Test
	void shouldClampCoordsToImage() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.9f, -20f, -10f, 600f, 700f, 1f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		assertThat(out.get(0).getBoundingBox()).containsExactly(0, 0, 500, 500);
	}

	@Test
	void shouldFilterByPerClassThreshold() {
		Map<Integer, Float> perClass = new HashMap<>();
		perClass.put(LayoutLabel.IMAGE.getIndex(), 0.9f);
		LayoutConfig config = LayoutConfig.builder()
			.scoreThreshold(0.5f)
			.classScoreThresholds(perClass)
			.build();
		LayoutPostProcessor pp = new LayoutPostProcessor(config);
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.7f, 0f, 0f, 50f, 50f, 1f),
			box(LayoutLabel.IMAGE, 0.55f, 0f, 0f, 50f, 50f, 2f),
			box(LayoutLabel.IMAGE, 0.95f, 60f, 60f, 110f, 110f, 3f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 200, 200);
		// 输出恒按 score 降序
		assertThat(out).extracting(LayoutResult::getLabel)
			.containsExactly(LayoutLabel.IMAGE, LayoutLabel.TEXT);
	}

	@Test
	void shouldSuppressOverlappingSameClassBoxesButKeepDifferentClass() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.9f, 0f, 0f, 100f, 100f, 1f),
			box(LayoutLabel.TEXT, 0.8f, 5f, 5f, 105f, 105f, 2f),
			box(LayoutLabel.TABLE, 0.7f, 5f, 5f, 105f, 105f, 3f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		assertThat(out).extracting(LayoutResult::getLabel)
			.containsExactly(LayoutLabel.TEXT, LayoutLabel.TABLE);
	}

	@Test
	void shouldKeepOverlappingSameClassBoxesWhenNmsDisabled() {
		LayoutConfig config = LayoutConfig.builder().layoutNms(false).build();
		LayoutPostProcessor pp = new LayoutPostProcessor(config);
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.9f, 0f, 0f, 100f, 100f, 1f),
			box(LayoutLabel.TEXT, 0.8f, 5f, 5f, 105f, 105f, 2f)
		};
		assertThat(pp.postProcess(boxes, 1d, 0, 0, 500, 500)).hasSize(2);
	}

	@Test
	void shouldDropOverlargeImageBoxButKeepText() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			box(LayoutLabel.IMAGE, 0.9f, 0f, 0f, 990f, 990f, 1f),
			box(LayoutLabel.IMAGE, 0.8f, 0f, 0f, 500f, 500f, 2f),
			box(LayoutLabel.TEXT, 0.7f, 0f, 0f, 1000f, 500f, 3f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 1000, 1000);
		assertThat(out).extracting(LayoutResult::getLabel)
			.containsExactly(LayoutLabel.IMAGE, LayoutLabel.TEXT);
		assertThat(out.get(0).getBoundingBox()).containsExactly(0, 0, 500, 500);
	}

	@Test
	void shouldRankReadingOrderByOrderColumn() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			box(LayoutLabel.PARAGRAPH_TITLE, 0.9f, 0f, 0f, 100f, 50f, 30f),
			box(LayoutLabel.TEXT, 0.85f, 0f, 100f, 100f, 150f, 10f),
			box(LayoutLabel.CONTENT, 0.8f, 0f, 200f, 100f, 250f, 20f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		// 输出按 score 降序：[PARAGRAPH_TITLE, TEXT, CONTENT]
		// order 键 TEXT(10) < CONTENT(20) < PARAGRAPH_TITLE(30) ⇒ 1-based 编号 1/2/3
		assertThat(out).extracting(LayoutResult::getReadingOrder)
			.containsExactly(3, 1, 2);
	}

	@Test
	void shouldBreakOrderTiesByScore() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.6f, 0f, 0f, 100f, 50f, 5f),
			box(LayoutLabel.CONTENT, 0.9f, 0f, 100f, 100f, 150f, 5f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		// 输出按 score 降序 [CONTENT, TEXT]；order 键相同 ⇒ 高分先读 ⇒ 1 / 2
		assertThat(out).extracting(LayoutResult::getReadingOrder).containsExactly(1, 2);
	}

	@Test
	void shouldSkipOrderLabelsWithoutConsumingIndex() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			box(LayoutLabel.DOC_TITLE, 0.9f, 0f, 0f, 400f, 50f, 1f),
			box(LayoutLabel.TABLE, 0.85f, 0f, 100f, 400f, 300f, 2f),
			box(LayoutLabel.TEXT, 0.8f, 0f, 350f, 400f, 450f, 3f),
			box(LayoutLabel.IMAGE, 0.75f, 0f, 500f, 400f, 700f, 4f),
			box(LayoutLabel.CONTENT, 0.7f, 0f, 750f, 400f, 850f, 5f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 1000);
		// TABLE / IMAGE 在 PaddleX SKIP_ORDER_LABELS 内 ⇒ 不占号，固定 -1
		// DOC_TITLE / TEXT / CONTENT 依次得 1 / 2 / 3
		assertThat(out).extracting(LayoutResult::getReadingOrder)
			.containsExactly(1, LayoutResult.NO_READING_ORDER, 2,
				LayoutResult.NO_READING_ORDER, 3);
	}

	@Test
	void shouldTreatAllLabelsAsOrderedWhenSkipListEmpty() {
		LayoutConfig config = LayoutConfig.builder()
			.skipOrderLabels(java.util.Collections.<LayoutLabel>emptySet())
			.build();
		LayoutPostProcessor pp = new LayoutPostProcessor(config);
		float[][] boxes = new float[][]{
			box(LayoutLabel.TABLE, 0.9f, 0f, 0f, 400f, 300f, 1f),
			box(LayoutLabel.TEXT, 0.8f, 0f, 350f, 400f, 450f, 2f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		// 空名单 ⇒ 全部参与编号
		assertThat(out).extracting(LayoutResult::getReadingOrder).containsExactly(1, 2);
	}

	@Test
	void shouldDegradeToScoreOrderWhenOrderColumnMissing() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			{LayoutLabel.TEXT.getIndex(), 0.9f, 0f, 0f, 100f, 50f},
			{LayoutLabel.TABLE.getIndex(), 0.8f, 0f, 100f, 100f, 150f}
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		assertThat(out).hasSize(2);
		// 缺 order 列 ⇒ 退化为按 score 降序的 1-based rank；TABLE 仍是跳过类 ⇒ -1
		assertThat(out).extracting(LayoutResult::getReadingOrder)
			.containsExactly(1, LayoutResult.NO_READING_ORDER);
	}

	@Test
	void shouldRespectMaxDetections() {
		LayoutConfig config = LayoutConfig.builder().maxDetections(2).build();
		LayoutPostProcessor pp = new LayoutPostProcessor(config);
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.9f, 0f, 0f, 50f, 50f, 1f),
			box(LayoutLabel.TEXT, 0.8f, 60f, 0f, 110f, 50f, 2f),
			box(LayoutLabel.TEXT, 0.7f, 120f, 0f, 170f, 50f, 3f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		assertThat(out).hasSize(2);
		assertThat(out).extracting(LayoutResult::getScore).containsExactly(0.9f, 0.8f);
	}

	@Test
	void shouldSkipUnknownClassAndDegenerateBox() {
		LayoutPostProcessor pp = new LayoutPostProcessor(LayoutConfig.builder().build());
		float[][] boxes = new float[][]{
			{99f, 0.9f, 0f, 0f, 100f, 100f, 1f},
			box(LayoutLabel.TEXT, 0.9f, 10f, 10f, 10f, 10f, 2f),
			box(LayoutLabel.TEXT, 0.9f, 0f, 0f, 100f, 100f, 3f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		assertThat(out).hasSize(1);
		assertThat(Arrays.toString(out.get(0).getBoundingBox())).isEqualTo("[0, 0, 100, 100]");
	}

	@Test
	void shouldApplyRelativeFloorAgainstTop1Score() {
		LayoutConfig config = LayoutConfig.builder()
			.scoreThreshold(0.2f)
			.scoreRatio(0.6f)
			.build();
		LayoutPostProcessor pp = new LayoutPostProcessor(config);
		// top1 = 0.90 ⇒ 相对下限 0.54；0.50 应被丢弃、0.60 应保留
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.90f, 0f, 0f, 100f, 50f, 1f),
			box(LayoutLabel.TEXT, 0.60f, 0f, 100f, 100f, 150f, 2f),
			box(LayoutLabel.TEXT, 0.50f, 0f, 200f, 100f, 250f, 3f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		assertThat(out).extracting(LayoutResult::getScore).containsExactly(0.90f, 0.60f);
	}

	@Test
	void shouldNotApplyRelativeFloorWhenDisabled() {
		LayoutConfig config = LayoutConfig.builder()
			.scoreThreshold(0.2f)
			.scoreRatio(0f)
			.build();
		LayoutPostProcessor pp = new LayoutPostProcessor(config);
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.90f, 0f, 0f, 100f, 50f, 1f),
			box(LayoutLabel.TEXT, 0.50f, 0f, 200f, 100f, 250f, 3f)
		};
		// scoreRatio=0 ⇒ 只走绝对阈值 0.2，低分框保留
		assertThat(pp.postProcess(boxes, 1d, 0, 0, 500, 500)).hasSize(2);
	}

	@Test
	void shouldLetAbsoluteThresholdWinOverRelativeFloor() {
		LayoutConfig config = LayoutConfig.builder()
			.scoreThreshold(0.8f)
			.scoreRatio(0.5f)
			.build();
		LayoutPostProcessor pp = new LayoutPostProcessor(config);
		// top1=0.9 ⇒ 相对下限 0.45，但绝对阈值 0.8 更高 ⇒ 取 0.8
		float[][] boxes = new float[][]{
			box(LayoutLabel.TEXT, 0.90f, 0f, 0f, 100f, 50f, 1f),
			box(LayoutLabel.TEXT, 0.70f, 0f, 100f, 100f, 150f, 2f)
		};
		List<LayoutResult> out = pp.postProcess(boxes, 1d, 0, 0, 500, 500);
		assertThat(out).extracting(LayoutResult::getScore).containsExactly(0.90f);
	}

	@Test
	void iouShouldBeSymmetric() {
		int[] a = new int[]{0, 0, 100, 100};
		int[] b = new int[]{50, 50, 150, 150};
		assertThat(LayoutPostProcessor.iou(a, b))
			.isEqualTo(LayoutPostProcessor.iou(b, a))
			.isCloseTo(1f / 7f, org.assertj.core.data.Offset.offset(1e-6f));
		assertThat(LayoutPostProcessor.iou(a, new int[]{200, 200, 300, 300})).isZero();
	}
}
