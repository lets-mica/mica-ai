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
package net.dreamlu.mica.ai.classify;

import net.dreamlu.mica.ai.classify.config.ClassifyConfig;
import net.dreamlu.mica.ai.classify.core.ClassifyEngine;
import net.dreamlu.mica.ai.classify.model.ClassifyResult;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.Mat;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 真实模型集成测试：加载 ONNX 跑完整推理，兼作 I/O 契约回归。
 *
 * <p>模型位于 {@code model-tools/classify/models/}，文件缺失时整类跳过
 * （不阻塞新 clone 仓库的 {@code mvn test}）。
 */
class ClassifyIntegrationTest {

	private static final String MODEL_DIR = "model-tools/classify/models";

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	/**
	 * 从当前工作目录向上回溯仓库根，定位 {@code model-tools} 下的文件；
	 * 模型支持系统属性 {@code mica.ai.classify.externalModel} 指定外置路径。
	 */
	private static File resolveModel(String name) {
		String external = System.getProperty("mica.ai.classify.externalModel");
		if (external != null) {
			File file = new File(external);
			return file.isFile() && file.length() > 1_000_000 ? file : null;
		}
		File file = resolve(MODEL_DIR + "/" + name + ".onnx");
		return file != null && file.length() > 1_000_000 ? file : null;
	}

	private static String resolveLabels(String name) {
		String external = System.getProperty("mica.ai.classify.externalLabels");
		if (external != null) {
			File file = new File(external);
			if (file.isFile()) {
				return file.getAbsolutePath();
			}
		}
		File file = resolve(MODEL_DIR + "/" + name + "_labels.txt");
		return file == null ? null : file.getAbsolutePath();
	}

	private static File resolve(String relative) {
		Path dir = Paths.get("").toAbsolutePath();
		while (dir != null) {
			File candidate = dir.resolve(relative).toFile();
			if (candidate.isFile()) {
				return candidate;
			}
			dir = dir.getParent();
		}
		return null;
	}

	private static byte[] loadImage(String resource) {
		try (InputStream in = Thread.currentThread().getContextClassLoader()
			.getResourceAsStream(resource)) {
			if (in == null) {
				throw new IllegalStateException("测试图片缺失: " + resource);
			}
			ByteArrayOutputStream out = new ByteArrayOutputStream();
			byte[] buffer = new byte[4096];
			int read;
			while ((read = in.read(buffer)) != -1) {
				out.write(buffer, 0, read);
			}
			return out.toByteArray();
		} catch (IOException e) {
			throw new UncheckedIOException(e);
		}
	}

	private static ClassifyEngine open(File model, String labels, String version,
										boolean multiLabel, int resizeShort) {
		ClassifyConfig config = ClassifyConfig.builder()
			.modelPath(model.getAbsolutePath())
			.modelVersion(version)
			.labelsPath(labels)
			.multiLabel(multiLabel)
			.resizeShort(resizeShort)
			.build();
		return ClassifyEngine.create(config);
	}

	@Test
	void textImageOrientationDetectsRotation() {
		File model = resolveModel("text_image_orientation");
		String labels = resolveLabels("text_image_orientation");
		Assumptions.assumeTrue(model != null && labels != null,
			"PULC text_image_orientation 模型/label 缺失，跳过");

		try (ClassifyEngine engine = open(model, labels, "text_image_orientation", false, 256)) {
			assertThat(engine.modelInputWidth()).isEqualTo(224);
			assertThat(engine.modelInputHeight()).isEqualTo(224);
			assertThat(engine.classCount()).isEqualTo(4);
			assertThat(engine.labels()).containsExactly("0", "90", "180", "270");
			assertThat(engine.isMultiLabel()).isFalse();

			assertThat(engine.classifyBytes(loadImage("text-orientation/rot0.jpg")).topLabel())
				.isEqualTo("0");
			assertThat(engine.classifyBytes(loadImage("text-orientation/rot180.jpg")).topLabel())
				.isEqualTo("180");
			assertThat(engine.classifyBytes(loadImage("text-orientation/rot90.jpg")).topLabel())
				.isEqualTo("90");
			assertThat(engine.classifyBytes(loadImage("text-orientation/rot270.jpg")).topLabel())
				.isEqualTo("270");
		}
	}

	@Test
	void clarityAssessmentSeparatesSharpAndBlur() {
		File model = resolveModel("clarity_assessment");
		String labels = resolveLabels("clarity_assessment");
		Assumptions.assumeTrue(model != null && labels != null,
			"PULC clarity_assessment 模型/label 缺失，跳过");

		try (ClassifyEngine engine = open(model, labels, "clarity_assessment", false, 256)) {
			assertThat(engine.classCount()).isEqualTo(2);
			assertThat(engine.labels()).containsExactly("clear", "blur");

			ClassifyResult sharp = engine.classifyBytes(loadImage("clarity/sharp.jpg"));
			ClassifyResult blur = engine.classifyBytes(loadImage("clarity/blur.jpg"));

			assertThat(sharp.topLabel()).isEqualTo("clear");
			assertThat(sharp.topScore()).isGreaterThan(0.6f);
			assertThat(blur.topLabel()).isEqualTo("blur");
			assertThat(blur.topScore()).isGreaterThan(0.9f);
		}
	}

	@Test
	void personExistsDetectsSomeone() {
		File model = resolveModel("person_exists");
		String labels = resolveLabels("person_exists");
		Assumptions.assumeTrue(model != null && labels != null,
			"PULC person_exists 模型/label 缺失，跳过");

		try (ClassifyEngine engine = open(model, labels, "person_exists", false, 256)) {
			assertThat(engine.classCount()).isEqualTo(2);
			assertThat(engine.labels()).containsExactly("nobody", "someone");

			ClassifyResult result = engine.classifyBytes(loadImage("person/someone.jpg"));

			assertThat(result.topLabel()).isEqualTo("someone");
			assertThat(result.topScore()).isGreaterThan(0.5f);
		}
	}

	@Test
	void tableAttributeScoresSixAttributes() {
		File model = resolveModel("table_attribute");
		String labels = resolveLabels("table_attribute");
		Assumptions.assumeTrue(model != null && labels != null,
			"PULC table_attribute 模型/label 缺失，跳过");

		// 官方配置直接 resize 到 224，无短边缩放 + 中心裁剪
		try (ClassifyEngine engine = open(model, labels, "table_attribute", true, 0)) {
			assertThat(engine.classCount()).isEqualTo(6);
			assertThat(engine.labels()).containsExactly(
				"source", "number", "color", "clarity", "obstruction", "angle");

			ClassifyResult table = engine.classifyBytes(loadImage("table/table.jpg"));
			ClassifyResult noTable = engine.classifyBytes(loadImage("table/no-table.jpg"));

			assertThat(table.classCount()).isEqualTo(6);
			assertThat(noTable.classCount()).isEqualTo(6);
			// 有表格图的 source 属性应当明显高于无表格图（相对比较，避免依赖绝对阈值）
			assertThat(table.scoreOf("source")).isGreaterThan(noTable.scoreOf("source"));
			assertThat(table.scoreOf("source")).isGreaterThan(0.5f);
			assertThat(noTable.scoreOf("source")).isLessThan(0.5f);
		}
	}

	@Test
	void rejectsLabelCountMismatch(@TempDir Path dir) throws IOException {
		File model = resolveModel("text_image_orientation");
		Assumptions.assumeTrue(model != null, "PULC text_image_orientation 模型缺失，跳过");

		Path labels = dir.resolve("bad-labels.txt");
		Files.write(labels, "0\n90\n180\n".getBytes(StandardCharsets.UTF_8));

		assertThatThrownBy(() -> open(model, labels.toString(),
			"text_image_orientation", false, 256))
			.isInstanceOf(net.dreamlu.mica.ai.common.exception.MicaAiException.class)
			.hasMessageContaining("不一致");
	}

	@Test
	void rejectsInputSizeMismatch() {
		File model = resolveModel("text_image_orientation");
		String labels = resolveLabels("text_image_orientation");
		Assumptions.assumeTrue(model != null && labels != null,
			"PULC text_image_orientation 模型/label 缺失，跳过");

		ClassifyConfig config = ClassifyConfig.builder()
			.modelPath(model.getAbsolutePath())
			.modelVersion("text_image_orientation")
			.labelsPath(labels)
			.inputWidth(192)
			.inputHeight(192)
			.resizeShort(224)
			.build();

		assertThatThrownBy(() -> ClassifyEngine.create(config).close())
			.isInstanceOf(net.dreamlu.mica.ai.common.exception.MicaAiException.class)
			.hasMessageContaining("inputHeight");
	}

	@Test
	void nullAndEmptyImageReturnNull() {
		File model = resolveModel("text_image_orientation");
		String labels = resolveLabels("text_image_orientation");
		Assumptions.assumeTrue(model != null && labels != null,
			"PULC text_image_orientation 模型/label 缺失，跳过");

		try (ClassifyEngine engine = open(model, labels, "text_image_orientation", false, 256)) {
			assertThat(engine.classify(null)).isNull();
			Mat empty = new Mat();
			try {
				assertThat(engine.classify(empty)).isNull();
			} finally {
				empty.release();
			}
		}
	}

	@Test
	void externalModelOverrideClassifiesImage() {
		String external = System.getProperty("mica.ai.classify.externalModel");
		Assumptions.assumeTrue(external != null, "未指定 mica.ai.classify.externalModel，跳过");

		File model = resolveModel("text_image_orientation");
		String labels = resolveLabels("text_image_orientation");
		Assumptions.assumeTrue(model != null && labels != null, "外置模型或 label 不可用，跳过");

		try (ClassifyEngine engine = open(model, labels, "text_image_orientation", false, 256)) {
			assertThat(engine.classifyBytes(loadImage("text-orientation/rot0.jpg")).topLabel())
				.isEqualTo("0");
		}
	}
}
