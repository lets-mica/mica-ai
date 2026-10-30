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
package net.dreamlu.mica.ai.humanseg;

import net.dreamlu.mica.ai.humanseg.config.HumanSegConfig;
import net.dreamlu.mica.ai.humanseg.core.HumanSegEngine;
import net.dreamlu.mica.ai.humanseg.util.HumanSegImageUtils;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.io.IOException;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * 真实模型集成测试：加载 ONNX 跑完整推理，兼作 I/O 契约回归。
 *
 * <p>模型位于 {@code model-tools/humanseg/models/}，文件缺失时整类跳过
 * （不阻塞新 clone 仓库的 {@code mvn test}）。
 */
class HumanSegIntegrationTest {

	private static final String MODEL =
		"model-tools/humanseg/models/human_segmentation_pphumanseg_2023mar.onnx";

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	/**
	 * 从当前工作目录向上回溯仓库根，定位 {@code model-tools} 下的模型；
	 * 支持系统属性 {@code mica.ai.humanseg.externalModel} 指定外置模型。
	 */
	private static File resolve(String relative) {
		String external = System.getProperty("mica.ai.humanseg.externalModel");
		if (external != null) {
			File file = new File(external);
			return file.isFile() && file.length() > 1_000_000 ? file : null;
		}
		java.nio.file.Path dir = Paths.get("").toAbsolutePath();
		while (dir != null) {
			File candidate = dir.resolve(relative).toFile();
			if (candidate.isFile() && candidate.length() > 1_000_000) {
				return candidate;
			}
			dir = dir.getParent();
		}
		return null;
	}

	private static byte[] loadSubjectImage() {
		try (InputStream in = Thread.currentThread().getContextClassLoader()
			.getResourceAsStream("subject.png")) {
			if (in == null) {
				throw new IllegalStateException("测试图片 subject.png 缺失");
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

	@Test
	void loadsBundledModelAndSegmentsSubject() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "PP-HumanSeg 模型缺失，跳过");

		byte[] image = loadSubjectImage();
		HumanSegConfig config = HumanSegConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();

		try (HumanSegEngine engine = HumanSegEngine.create(config)) {
			assertThat(engine.modelInputWidth()).isEqualTo(192);
			assertThat(engine.modelInputHeight()).isEqualTo(192);
			assertThat(engine.outputCount()).isEqualTo(1);

			Mat alpha = engine.alphaBytes(image);
			try {
				// 原图 480×360，掩码必须回到原尺寸
				assertThat(alpha.cols()).isEqualTo(480);
				assertThat(alpha.rows()).isEqualTo(360);
				// 主体中心（实测 ≈0.996）应当是前景，四角（实测 <0.01）应当是背景
				assertThat(alpha.get(200, 240)[0]).isCloseTo(0.996, within(0.01));
				assertThat(alpha.get(10, 10)[0]).isLessThan(0.05);
				assertThat(alpha.get(350, 10)[0]).isLessThan(0.05);
			} finally {
				alpha.release();
			}
		}
	}

	@Test
	void cutoutBytesProducesTransparentPng() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "PP-HumanSeg 模型缺失，跳过");

		byte[] image = loadSubjectImage();
		HumanSegConfig config = HumanSegConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();

		try (HumanSegEngine engine = HumanSegEngine.create(config)) {
			byte[] png = engine.cutoutBytes(image);
			assertThat(png).isNotEmpty();

			Mat decoded = decode(png, Imgcodecs.IMREAD_UNCHANGED);
			try {
				assertThat(decoded.channels()).isEqualTo(4);
				assertThat(decoded.cols()).isEqualTo(480);
				assertThat(decoded.rows()).isEqualTo(360);
				// 主体中心的不透明通道应为全不透明（alpha≈1 → 255）
				assertThat(decoded.get(200, 240)[3]).isGreaterThan(250d);
				// 背景角落应为全透明（alpha≈0 → 0）
				assertThat(decoded.get(10, 10)[3]).isLessThan(15d);
			} finally {
				decoded.release();
			}
		}
	}

	@Test
	void maskBinaryBytesProducesSingleChannelMask() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "PP-HumanSeg 模型缺失，跳过");

		byte[] image = loadSubjectImage();
		HumanSegConfig config = HumanSegConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();

		try (HumanSegEngine engine = HumanSegEngine.create(config)) {
			byte[] png = engine.maskBinaryBytes(image);
			Mat decoded = decode(png, Imgcodecs.IMREAD_UNCHANGED);
			try {
				assertThat(decoded.channels()).isEqualTo(1);
				assertThat(decoded.get(200, 240)[0]).isEqualTo(255d);
				assertThat(decoded.get(10, 10)[0]).isEqualTo(0d);
			} finally {
				decoded.release();
			}
		}
	}

	@Test
	void rejectsMismatchedInputSize() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "PP-HumanSeg 模型缺失，跳过");

		HumanSegConfig config = HumanSegConfig.builder()
			.modelPath(model.getAbsolutePath())
			.inputWidth(256)
			.inputHeight(144)
			.build();

		assertThatThrownBy(() -> HumanSegEngine.create(config).close())
			.hasMessageContaining("inputWidth/inputHeight");
	}

	@Test
	void emptyImageReturnsNullAlpha() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "PP-HumanSeg 模型缺失，跳过");

		HumanSegConfig config = HumanSegConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();
		try (HumanSegEngine engine = HumanSegEngine.create(config)) {
			assertThat(engine.alpha((Mat) null)).isNull();
			Mat empty = new Mat();
			try {
				assertThat(engine.alpha(empty)).isNull();
			} finally {
				empty.release();
			}
		}
	}

	private static Mat decode(byte[] bytes, int flags) {
		MatOfByte mob = new MatOfByte(bytes);
		try {
			Mat mat = Imgcodecs.imdecode(mob, flags);
			assertThat(mat.empty()).isFalse();
			return mat;
		} finally {
			HumanSegImageUtils.releaseAll(mob);
		}
	}
}
