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
package net.dreamlu.mica.ai.docalign;

import net.dreamlu.mica.ai.docalign.config.DocAlignConfig;
import net.dreamlu.mica.ai.docalign.model.DocAlignResult;
import net.dreamlu.mica.ai.docalign.model.DocCorner;
import net.dreamlu.mica.ai.docalign.util.DocAlignImageUtils;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * 真实模型集成测试：加载 ONNX 跑完整推理，兼作 I/O 契约回归。
 *
 * <p>测试图 {@code doc_warp.png} 是<b>合成</b>的斜拍文档（800×600），
 * 文档四角真值（像素坐标）：左上 (150,90) / 右上 (690,60) / 右下 (730,520) / 左下 (110,540)。
 * 该图同时用于验证「4 个热力图通道的角点顺序」这一最容易写反的契约。
 */
class DocAlignIntegrationTest {

	private static final String MODEL =
		"model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx";

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	private static File resolve(String relative) {
		String external = System.getProperty("mica.ai.docalign.externalModel");
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

	private static byte[] loadDocImage() {
		try (InputStream in = Thread.currentThread().getContextClassLoader()
			.getResourceAsStream("doc_warp.png")) {
			if (in == null) {
				throw new IllegalStateException("测试图片 doc_warp.png 缺失");
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
	void loadsBundledModelAndFindsCornersInExpectedOrder() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "DocAligner 模型缺失，跳过");

		DocAlignConfig config = DocAlignConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();
		try (DocAlignEngine engine = DocAlignEngine.create(config)) {
			assertThat(engine.modelInputWidth()).isEqualTo(256);
			assertThat(engine.modelInputHeight()).isEqualTo(256);
			assertThat(engine.outputCount()).isEqualTo(1);

			DocAlignResult result = engine.cornersBytes(loadDocImage());
			assertThat(result.getWidth()).isEqualTo(800);
			assertThat(result.getHeight()).isEqualTo(600);
			assertThat(result.getCorners()).hasSize(4);

			// 通道顺序：左上 → 右上 → 右下 → 左下（实测偏差约 10px，热力图质心略向内收）
			assertCorner(result.getTopLeft(), 150, 90);
			assertCorner(result.getTopRight(), 690, 60);
			assertCorner(result.getBottomRight(), 730, 520);
			assertCorner(result.getBottomLeft(), 110, 540);

			assertThat(result.getTopLeft().getScore()).isGreaterThan(0.5);
			assertThat(result.getBottomRight().getScore()).isGreaterThan(0.5);
		}
	}

	@Test
	void alignBytesFlattensDocument() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "DocAligner 模型缺失，跳过");

		DocAlignConfig config = DocAlignConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();
		try (DocAlignEngine engine = DocAlignEngine.create(config)) {
			byte[] png = engine.alignBytes(loadDocImage());
			assertThat(png).isNotEmpty();

			Mat decoded = decode(png);
			try {
				int[] expected = DocAlignImageUtils.suggestOutputSize(
					engine.cornersBytes(loadDocImage()));
				assertThat(decoded.cols()).isEqualTo(expected[0]);
				assertThat(decoded.rows()).isEqualTo(expected[1]);
				// 拉平后的画面应该以白色纸面为主（合成图是「白底 + 黑线条」）
				assertThat(Core.mean(decoded).val[0]).isGreaterThan(180d);
			} finally {
				decoded.release();
			}
		}
	}

	@Test
	void rejectsMismatchedInputSize() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "DocAligner 模型缺失，跳过");

		DocAlignConfig config = DocAlignConfig.builder()
			.modelPath(model.getAbsolutePath())
			.inputWidth(320)
			.inputHeight(320)
			.build();

		assertThatThrownBy(() -> DocAlignEngine.create(config).close())
			.hasMessageContaining("inputWidth/inputHeight");
	}

	@Test
	void emptyImageReturnsNullCorners() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "DocAligner 模型缺失，跳过");

		DocAlignConfig config = DocAlignConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();
		try (DocAlignEngine engine = DocAlignEngine.create(config)) {
			assertThat(engine.corners((Mat) null)).isNull();
			Mat empty = new Mat();
			try {
				assertThat(engine.corners(empty)).isNull();
			} finally {
				empty.release();
			}
		}
	}

	private static void assertCorner(DocCorner corner, double expectedX, double expectedY) {
		// 热力图质心天然向文档内收，实测偏差 10~23px（角点间距 >400px，30px 容差不会混淆顺序）
		assertThat(corner.getX()).isCloseTo(expectedX, within(30d));
		assertThat(corner.getY()).isCloseTo(expectedY, within(30d));
	}

	private static Mat decode(byte[] bytes) {
		MatOfByte mob = new MatOfByte(bytes);
		try {
			Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
			assertThat(mat.empty()).isFalse();
			return mat;
		} finally {
			DocAlignImageUtils.releaseAll(mob);
		}
	}
}
