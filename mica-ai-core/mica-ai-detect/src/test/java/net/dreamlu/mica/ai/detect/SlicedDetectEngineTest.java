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
package net.dreamlu.mica.ai.detect;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.detect.config.DetectConfig;
import net.dreamlu.mica.ai.detect.config.DetectModelFamily;
import net.dreamlu.mica.ai.detect.config.SlicedDetectConfig;
import net.dreamlu.mica.ai.detect.core.DetectSource;
import net.dreamlu.mica.ai.detect.core.SlicedDetectEngine;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import net.dreamlu.mica.ai.detect.model.SliceMergeStrategy;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link SlicedDetectEngine} 切片编排测试（用假检测源，零 ONNX 依赖）。
 *
 * <p>覆盖切片网格 → 坐标回映射 → 跨片合并这条主链路：
 * 切片边长恒等于模型输入边长、每片都是正方形、框被平移回原图坐标、
 * 跨片重复框被合并、越界框被裁剪。
 */
class SlicedDetectEngineTest {

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	/**
	 * 假检测源：按调用顺序返回预设的框，并记录每次收到的切片尺寸。
	 */
	private static final class FakeDetectSource implements DetectSource {

		private final List<List<DetectBox>> responses = new ArrayList<>();
		private final List<int[]> tileSizes = new ArrayList<>();
		private int index;

		FakeDetectSource respondWith(DetectBox... boxes) {
			responses.add(Arrays.asList(boxes));
			return this;
		}

		@Override
		public DetectResult detect(Mat bgr) {
			tileSizes.add(new int[]{bgr.cols(), bgr.rows()});
			List<DetectBox> boxes = index < responses.size() ? responses.get(index)
				: Collections.emptyList();
			index++;
			return new DetectResult(bgr.cols(), bgr.rows(), boxes);
		}

	}

	private static DetectConfig detectConfig() {
		return DetectConfig.builder()
			.modelPath("model-tools/detect/models/object_detection_nanodet_2022nov.onnx")
			.family(DetectModelFamily.NANODET_PLUS)
			.inputSize(416)
			.build();
	}

	private static SlicedDetectConfig slicedConfig(SliceMergeStrategy strategy) {
		return SlicedDetectConfig.builder()
			.detectConfig(detectConfig())
			.overlapRatio(0.25f)
			.mergeStrategy(strategy)
			.build();
	}

	private static Mat blankImage(int width, int height) {
		return new Mat(height, width, CvType.CV_8UC3);
	}

	@Test
	void everyTileIsSquareOfModelInputSize() {
		FakeDetectSource source = new FakeDetectSource();
		SlicedDetectEngine engine = SlicedDetectEngine.create(source, slicedConfig(SliceMergeStrategy.KEEP_ALL));

		Mat image = blankImage(800, 800);
		try {
			engine.detect(image);
		} finally {
			image.release();
		}

		assertThat(source.tileSizes).isNotEmpty();
		for (int[] size : source.tileSizes) {
			assertThat(size[0]).isEqualTo(416);
			assertThat(size[1]).isEqualTo(416);
		}
	}

	@Test
	void mapsSliceBoxesBackToOriginalCoordinates() {
		FakeDetectSource source = new FakeDetectSource();
		for (int i = 0; i < 9; i++) {
			source.respondWith(new DetectBox(10f, 10f, 50f, 50f, 0.9f, 0));
		}
		SlicedDetectEngine engine = SlicedDetectEngine.create(source, slicedConfig(SliceMergeStrategy.KEEP_ALL));

		Mat image = blankImage(800, 800);
		try {
			DetectResult result = engine.detect(image);

			assertThat(result.getWidth()).isEqualTo(800);
			assertThat(result.getHeight()).isEqualTo(800);
			assertThat(result.getBoxes()).hasSize(9);

			List<String> positions = new ArrayList<>();
			for (DetectBox box : result.getBoxes()) {
				positions.add(Math.round(box.getX1()) + "," + Math.round(box.getY1()));
			}
			// 800 宽在 416 切片 / 312 步进下的起点是 0 / 312 / 384
			assertThat(positions).containsExactlyInAnyOrder(
				"10,10", "322,10", "394,10",
				"10,322", "322,322", "394,322",
				"10,394", "322,394", "394,394");
		} finally {
			image.release();
		}
	}

	@Test
	void mergesDuplicateBoxesAcrossSlices() {
		FakeDetectSource source = new FakeDetectSource();
		source.respondWith(new DetectBox(300f, 10f, 380f, 90f, 0.9f, 0));
		source.respondWith(new DetectBox(0f, 10f, 80f, 90f, 0.8f, 0));
		SlicedDetectEngine engine = SlicedDetectEngine.create(source, slicedConfig(SliceMergeStrategy.IOU_MERGE));

		Mat image = blankImage(800, 100);
		try {
			DetectResult result = engine.detect(image);

			// 两片的框在原图坐标系里 IoU 约 0.74 > nmsThreshold 0.6 → 只保留高分框
			assertThat(result.getBoxes()).hasSize(1);
			assertThat(result.getBoxes().get(0).getScore()).isEqualTo(0.9f);
			assertThat(result.getBoxes().get(0).getX1()).isEqualTo(300f);
		} finally {
			image.release();
		}
	}

	@Test
	void keepsAllBoxesWhenStrategyIsKeepAll() {
		FakeDetectSource source = new FakeDetectSource();
		source.respondWith(new DetectBox(300f, 10f, 380f, 90f, 0.9f, 0));
		source.respondWith(new DetectBox(0f, 10f, 80f, 90f, 0.8f, 0));
		SlicedDetectEngine engine = SlicedDetectEngine.create(source, slicedConfig(SliceMergeStrategy.KEEP_ALL));

		Mat image = blankImage(800, 100);
		try {
			DetectResult result = engine.detect(image);

			assertThat(result.getBoxes()).hasSize(2);
		} finally {
			image.release();
		}
	}

	@Test
	void clipsBoxesToOriginalBounds() {
		FakeDetectSource source = new FakeDetectSource();
		// 800 宽下有 3 片（起点 0 / 312 / 384），让最后一片（起点 384）产生越界框
		source.respondWith();
		source.respondWith();
		source.respondWith(new DetectBox(400f, 10f, 500f, 90f, 0.9f, 0));
		SlicedDetectEngine engine = SlicedDetectEngine.create(source, slicedConfig(SliceMergeStrategy.KEEP_ALL));

		Mat image = blankImage(800, 100);
		try {
			DetectResult result = engine.detect(image);

			assertThat(result.getBoxes()).hasSize(1);
			DetectBox box = result.getBoxes().get(0);
			assertThat(box.getX1()).isEqualTo(784f);
			assertThat(box.getX2()).isEqualTo(800f);
		} finally {
			image.release();
		}
	}

	@Test
	void throwsWhenSliceCountExceedsLimit() {
		FakeDetectSource source = new FakeDetectSource();
		SlicedDetectConfig config = SlicedDetectConfig.builder()
			.detectConfig(detectConfig())
			.maxSlices(4)
			.build();
		SlicedDetectEngine engine = SlicedDetectEngine.create(source, config);

		Mat image = blankImage(2000, 2000);
		try {
			assertThatThrownBy(() -> engine.detect(image))
				.isInstanceOf(MicaAiException.class)
				.hasMessageContaining("片");
		} finally {
			image.release();
		}
	}

	@Test
	void returnsNullForEmptyImage() {
		FakeDetectSource source = new FakeDetectSource();
		SlicedDetectEngine engine = SlicedDetectEngine.create(source, slicedConfig(SliceMergeStrategy.IOU_MERGE));

		Mat empty = new Mat();
		try {
			assertThat(engine.detect(empty)).isNull();
			assertThat(engine.detect((Mat) null)).isNull();
		} finally {
			empty.release();
		}
	}

	@Test
	void reportsSliceCountWithoutInference() {
		SlicedDetectEngine engine = SlicedDetectEngine.create(new FakeDetectSource(),
			slicedConfig(SliceMergeStrategy.IOU_MERGE));

		assertThat(engine.sliceCount(1200, 1500)).isEqualTo(20);
		assertThat(engine.sliceCount(300, 200)).isEqualTo(1);
	}

}
