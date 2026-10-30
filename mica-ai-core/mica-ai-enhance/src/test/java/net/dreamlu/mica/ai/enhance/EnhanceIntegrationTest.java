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
package net.dreamlu.mica.ai.enhance;

import net.dreamlu.mica.ai.enhance.config.EnhanceConfig;
import net.dreamlu.mica.ai.enhance.util.EnhanceImageUtils;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 真实模型集成测试：加载 ONNX 跑完整推理，兼作 I/O 契约回归。
 *
 * <p>模型位于 {@code model-tools/enhance/models/}，文件缺失时整类跳过
 * （不阻塞新 clone 仓库的 {@code mvn test}）。
 */
class EnhanceIntegrationTest {

	private static final String MODEL =
		"model-tools/enhance/models/realesr-general-x4v3.onnx";

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	/**
	 * 从当前工作目录向上回溯仓库根，定位 {@code model-tools} 下的模型；
	 * 支持系统属性 {@code mica.ai.enhance.externalModel} 指定外置模型。
	 */
	private static File resolve(String relative) {
		String external = System.getProperty("mica.ai.enhance.externalModel");
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
	void loadsBundledModelAndUpscalesFourTimes() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "Real-ESRGAN 模型缺失，跳过");

		byte[] image = loadSubjectImage();
		EnhanceConfig config = EnhanceConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();

		try (EnhanceEngine engine = EnhanceEngine.create(config)) {
			assertThat(engine.outputCount()).isEqualTo(1);

			byte[] png = engine.upscaleBytes(image);
			assertThat(png).isNotEmpty();

			Mat decoded = decode(png);
			Mat source = EnhanceImageUtils.byteArrayToMat(image);
			Mat reference = new Mat();
			try {
				// 原图 480×360 → 4 倍后 1920×1440（该尺寸超过 tileSize，覆盖了 tile 拼接路径）
				assertThat(decoded.cols()).isEqualTo(1920);
				assertThat(decoded.rows()).isEqualTo(1440);
				assertThat(decoded.channels()).isEqualTo(3);

				// 与双三次放大对比：超分结果应明显接近参考画面（不是黑图 / 花色噪声）
				Imgproc.resize(source, reference,
					new Size(decoded.cols(), decoded.rows()), 0, 0, Imgproc.INTER_CUBIC);
				assertThat(meanAbsDiff(decoded, reference)).isLessThan(0.06);

				// 中心区域不应是纯黑
				double[] center = decoded.get(720, 960);
				assertThat(center[0] + center[1] + center[2]).isGreaterThan(30d);
			} finally {
				decoded.release();
				source.release();
				reference.release();
			}
		}
	}

	@Test
	void smallImageUsesSingleTilePath() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "Real-ESRGAN 模型缺失，跳过");

		Mat source = EnhanceImageUtils.byteArrayToMat(loadSubjectImage());
		Mat small = null;
		Mat upscaled = null;
		EnhanceConfig config = EnhanceConfig.builder()
			.modelPath(model.getAbsolutePath())
			.tileSize(256)
			.build();
		try (EnhanceEngine engine = EnhanceEngine.create(config)) {
			small = EnhanceImageUtils.crop(source, 0, 0, 200, 150);
			upscaled = engine.upscale(small);
			assertThat(upscaled.cols()).isEqualTo(800);
			assertThat(upscaled.rows()).isEqualTo(600);
		} finally {
			source.release();
			EnhanceImageUtils.releaseAll(small, upscaled);
		}
	}

	@Test
	void emptyImageReturnsNull() {
		File model = resolve(MODEL);
		Assumptions.assumeTrue(model != null, "Real-ESRGAN 模型缺失，跳过");

		EnhanceConfig config = EnhanceConfig.builder()
			.modelPath(model.getAbsolutePath())
			.build();
		try (EnhanceEngine engine = EnhanceEngine.create(config)) {
			assertThat(engine.upscale((Mat) null)).isNull();
			Mat empty = new Mat();
			try {
				assertThat(engine.upscale(empty)).isNull();
			} finally {
				empty.release();
			}
		}
	}

	private static double meanAbsDiff(Mat a, Mat b) {
		byte[] pa = new byte[a.channels() * a.cols() * a.rows()];
		byte[] pb = new byte[b.channels() * b.cols() * b.rows()];
		a.get(0, 0, pa);
		b.get(0, 0, pb);
		long sum = 0;
		for (int i = 0; i < pa.length; i++) {
			sum += Math.abs((pa[i] & 0xff) - (pb[i] & 0xff));
		}
		return sum / (255d * pa.length);
	}

	private static Mat decode(byte[] bytes) {
		MatOfByte mob = new MatOfByte(bytes);
		try {
			Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
			assertThat(mat.empty()).isFalse();
			return mat;
		} finally {
			mob.release();
		}
	}
}
