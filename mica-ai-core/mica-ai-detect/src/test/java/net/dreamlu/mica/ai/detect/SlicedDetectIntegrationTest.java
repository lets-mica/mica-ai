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
import net.dreamlu.mica.ai.detect.core.DetectEngine;
import net.dreamlu.mica.ai.detect.core.SlicedDetectEngine;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import net.dreamlu.mica.ai.detect.model.SliceMergeStrategy;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.File;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 切片推理的真实模型集成测试：加载 ONNX 跑完整切片推理，兼作 I/O 契约回归。
 *
 * <p>模型来自 opencv_zoo（Apache-2.0），位于 {@code model-tools/detect/models/}；
 * 文件缺失时整类跳过（不阻塞新 clone 仓库的 {@code mvn test}）。
 *
 * <p>「切片比整图多检出小目标」这一效果依赖真实拍摄画面，由 example 端点
 * {@code POST /detect/sliced-compare} 在实拍图上量化；本类验证的是切片管线的
 * 几何与契约正确性（坐标回原图、边界裁剪、片数上限、合并策略）。
 */
class SlicedDetectIntegrationTest {

	private static final String NANODET_MODEL =
		"model-tools/detect/models/object_detection_nanodet_2022nov.onnx";

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	private static File resolve(String relative) {
		String external = System.getProperty("mica.ai.detect.externalModel");
		if (external != null) {
			File file = new File(external);
			return file.isFile() && file.length() > 1_000_000 ? file : null;
		}
		java.nio.file.Path dir = java.nio.file.Paths.get("").toAbsolutePath();
		while (dir != null) {
			File candidate = dir.resolve(relative).toFile();
			if (candidate.isFile() && candidate.length() > 1_000_000) {
				return candidate;
			}
			dir = dir.getParent();
		}
		return null;
	}

	private static Mat syntheticImage(int width, int height) {
		Mat mat = new Mat(height, width, CvType.CV_8UC3);
		byte[] data = new byte[width * height * 3];
		Random random = new Random(20261126L);
		random.nextBytes(data);
		mat.put(0, 0, data);
		return mat;
	}

	private static SlicedDetectConfig slicedConfig(DetectConfig detectConfig, int maxSlices,
												   SliceMergeStrategy strategy) {
		return SlicedDetectConfig.builder()
			.detectConfig(detectConfig)
			.overlapRatio(0.25f)
			.maxSlices(maxSlices)
			.mergeStrategy(strategy)
			.build();
	}

	@Test
	void slicedInferenceProducesBoxesInOriginalCoordinates() {
		File model = resolve(NANODET_MODEL);
		Assumptions.assumeTrue(model != null, "NanoDet-Plus 模型缺失，跳过");

		DetectConfig detectConfig = DetectConfig.builder()
			.modelPath(model.getAbsolutePath())
			.family(DetectModelFamily.NANODET_PLUS)
			.build();

		Mat image = syntheticImage(1000, 800);
		try (DetectEngine engine = DetectEngine.create(detectConfig)) {
			SlicedDetectEngine sliced = SlicedDetectEngine.create(engine,
				slicedConfig(detectConfig, 64, SliceMergeStrategy.IOU_MERGE));

			DetectResult result = sliced.detect(image);

			assertThat(result).isNotNull();
			assertThat(result.getWidth()).isEqualTo(1000);
			assertThat(result.getHeight()).isEqualTo(800);
			float previousScore = Float.MAX_VALUE;
			for (DetectBox box : result.getBoxes()) {
				assertThat(box.getScore()).isBetween(0f, 1f);
				assertThat(box.getClassId()).isBetween(0, 79);
				// 切片结果必须被裁剪回原图范围内
				assertThat(box.getX1()).isBetween(0f, 1000f);
				assertThat(box.getX2()).isBetween(0f, 1000f);
				assertThat(box.getY1()).isBetween(0f, 800f);
				assertThat(box.getY2()).isBetween(0f, 800f);
				assertThat(box.getX2()).isGreaterThan(box.getX1());
				assertThat(box.getY2()).isGreaterThan(box.getY1());
				assertThat(box.getScore()).isLessThanOrEqualTo(previousScore);
				previousScore = box.getScore();
			}
		} finally {
			image.release();
		}
	}

	@Test
	void keepAllStrategyReturnsAtLeastAsManyBoxesAsMerge() {
		File model = resolve(NANODET_MODEL);
		Assumptions.assumeTrue(model != null, "NanoDet-Plus 模型缺失，跳过");

		DetectConfig detectConfig = DetectConfig.builder()
			.modelPath(model.getAbsolutePath())
			.family(DetectModelFamily.NANODET_PLUS)
			.build();

		Mat image = syntheticImage(900, 700);
		try (DetectEngine engine = DetectEngine.create(detectConfig)) {
			SlicedDetectEngine merged = SlicedDetectEngine.create(engine,
				slicedConfig(detectConfig, 64, SliceMergeStrategy.IOU_MERGE));
			SlicedDetectEngine kept = SlicedDetectEngine.create(engine,
				slicedConfig(detectConfig, 64, SliceMergeStrategy.KEEP_ALL));

			int mergedCount = merged.detect(image).size();
			int keptCount = kept.detect(image).size();

			assertThat(keptCount).isGreaterThanOrEqualTo(mergedCount);
		} finally {
			image.release();
		}
	}

	@Test
	void rejectsImageExceedingSliceLimit() {
		File model = resolve(NANODET_MODEL);
		Assumptions.assumeTrue(model != null, "NanoDet-Plus 模型缺失，跳过");

		DetectConfig detectConfig = DetectConfig.builder()
			.modelPath(model.getAbsolutePath())
			.family(DetectModelFamily.NANODET_PLUS)
			.build();

		Mat image = syntheticImage(2400, 2400);
		try (DetectEngine engine = DetectEngine.create(detectConfig)) {
			SlicedDetectEngine sliced = SlicedDetectEngine.create(engine,
				slicedConfig(detectConfig, 8, SliceMergeStrategy.IOU_MERGE));

			assertThatThrownBy(() -> sliced.detect(image))
				.isInstanceOf(MicaAiException.class)
				.hasMessageContaining("片");
		} finally {
			image.release();
		}
	}

	@Test
	void emptyImageReturnsNullResult() {
		File model = resolve(NANODET_MODEL);
		Assumptions.assumeTrue(model != null, "NanoDet-Plus 模型缺失，跳过");

		DetectConfig detectConfig = DetectConfig.builder()
			.modelPath(model.getAbsolutePath())
			.family(DetectModelFamily.NANODET_PLUS)
			.build();

		try (DetectEngine engine = DetectEngine.create(detectConfig)) {
			SlicedDetectEngine sliced = SlicedDetectEngine.create(engine,
				slicedConfig(detectConfig, 64, SliceMergeStrategy.IOU_MERGE));

			Mat empty = new Mat();
			try {
				assertThat(sliced.detect(empty)).isNull();
				assertThat(sliced.detect((Mat) null)).isNull();
			} finally {
				empty.release();
			}
		}
	}

}
