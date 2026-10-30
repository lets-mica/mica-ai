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

import net.dreamlu.mica.ai.detect.config.DetectConfig;
import net.dreamlu.mica.ai.detect.config.DetectModelFamily;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.File;
import java.util.Random;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 真实模型集成测试：加载 ONNX 跑完整推理，兼作 I/O 契约回归。
 *
 * <p>模型来自 opencv_zoo（Apache-2.0），位于 {@code model-tools/detect/models/}；
 * 文件缺失时整类跳过（不阻塞新 clone 仓库的 {@code mvn test}）。
 */
class DetectIntegrationTest {

	private static final String NANODET_MODEL =
		"model-tools/detect/models/object_detection_nanodet_2022nov.onnx";
	private static final String YOLOX_MODEL =
		"model-tools/detect/models/object_detection_yolox_2022nov.onnx";

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	/**
	 * 从当前工作目录向上回溯仓库根，定位 {@code model-tools} 下的模型；
	 * 支持系统属性 {@code mica.ai.detect.externalModel} 指定外置模型。
	 */
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
		Random random = new Random(20261030L);
		random.nextBytes(data);
		mat.put(0, 0, data);
		return mat;
	}

	private static void assertValidBoxes(DetectResult result, int width, int height, int numClasses) {
		assertThat(result).isNotNull();
		assertThat(result.getWidth()).isEqualTo(width);
		assertThat(result.getHeight()).isEqualTo(height);
		float previousScore = Float.MAX_VALUE;
		for (DetectBox box : result.getBoxes()) {
			assertThat(box.getScore()).isBetween(0f, 1f);
			assertThat(box.getClassId()).isBetween(0, numClasses - 1);
			assertThat(box.getX1()).isBetween(0f, (float) width);
			assertThat(box.getX2()).isBetween(0f, (float) width);
			assertThat(box.getY1()).isBetween(0f, (float) height);
			assertThat(box.getY2()).isBetween(0f, (float) height);
			assertThat(box.getX2()).isGreaterThan(box.getX1());
			assertThat(box.getY2()).isGreaterThan(box.getY1());
			assertThat(box.getScore()).isLessThanOrEqualTo(previousScore);
			previousScore = box.getScore();
		}
	}

	@Test
	void nanodetPlusLoadsAndDetects() {
		File model = resolve(NANODET_MODEL);
		Assumptions.assumeTrue(model != null, "NanoDet-Plus 模型缺失，跳过");

		DetectConfig config = DetectConfig.builder()
			.modelPath(model.getAbsolutePath())
			.family(DetectModelFamily.NANODET_PLUS)
			.build();

		Mat image = syntheticImage(640, 480);
		try (DetectEngine engine = DetectEngine.create(config)) {
			assertThat(engine.modelInputSize()).isEqualTo(416);
			assertThat(engine.outputCount()).isEqualTo(6);
			assertThat(engine.getLabels()).hasSize(80);

			DetectResult result = engine.detect(image);
			assertValidBoxes(result, 640, 480, 80);
		} finally {
			image.release();
		}
	}

	@Test
	void yoloxLoadsAndDetects() {
		File model = resolve(YOLOX_MODEL);
		Assumptions.assumeTrue(model != null, "YOLOX-s 模型缺失，跳过");

		DetectConfig config = DetectConfig.builder()
			.modelPath(model.getAbsolutePath())
			.family(DetectModelFamily.YOLOX_S)
			.inputSize(640)
			.scoreThreshold(0.5f)
			.nmsThreshold(0.5f)
			.build();

		Mat image = syntheticImage(800, 600);
		try (DetectEngine engine = DetectEngine.create(config)) {
			assertThat(engine.modelInputSize()).isEqualTo(640);
			assertThat(engine.outputCount()).isEqualTo(1);

			DetectResult result = engine.detect(image);
			assertValidBoxes(result, 800, 600, 80);
		} finally {
			image.release();
		}
	}

	@Test
	void nanodetRejectsMismatchedInputSize() {
		File model = resolve(NANODET_MODEL);
		Assumptions.assumeTrue(model != null, "NanoDet-Plus 模型缺失，跳过");

		DetectConfig config = DetectConfig.builder()
			.modelPath(model.getAbsolutePath())
			.inputSize(640)
			.build();

		org.assertj.core.api.Assertions.assertThatThrownBy(() -> DetectEngine.create(config).close())
			.hasMessageContaining("inputSize");
	}

	@Test
	void emptyImageReturnsNullResult() {
		File model = resolve(NANODET_MODEL);
		Assumptions.assumeTrue(model != null, "NanoDet-Plus 模型缺失，跳过");

		DetectConfig config = DetectConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();
		try (DetectEngine engine = DetectEngine.create(config)) {
			assertThat(engine.detect((Mat) null)).isNull();
			Mat empty = new Mat();
			try {
				assertThat(engine.detect(empty)).isNull();
			} finally {
				empty.release();
			}
		}
	}
}
