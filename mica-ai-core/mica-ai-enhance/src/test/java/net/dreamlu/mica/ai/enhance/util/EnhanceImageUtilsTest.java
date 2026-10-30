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
package net.dreamlu.mica.ai.enhance.util;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

class EnhanceImageUtilsTest {

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	@Test
	void convertsBgrToRgbChwWithDivideBy255() {
		Mat bgr = new Mat(1, 1, CvType.CV_8UC3);
		// B=10, G=20, R=30
		bgr.put(0, 0, new byte[]{10, 20, 30});
		try {
			float[] chw = EnhanceImageUtils.bgrToRgbChwFloat(bgr);
			assertThat(chw).hasSize(3);
			assertThat(chw[0]).isCloseTo(30f / 255f, within(1e-5f));
			assertThat(chw[1]).isCloseTo(20f / 255f, within(1e-5f));
			assertThat(chw[2]).isCloseTo(10f / 255f, within(1e-5f));
		} finally {
			bgr.release();
		}
	}

	@Test
	void chwLengthMatchesThreePlanes() {
		Mat bgr = new Mat(3, 4, CvType.CV_8UC3);
		try {
			assertThat(EnhanceImageUtils.bgrToRgbChwFloat(bgr)).hasSize(3 * 4 * 3);
		} finally {
			bgr.release();
		}
	}

	@Test
	void convertsChwBackToBgrWithChannelOrder() {
		// R=0.4, G=0.6, B=0.8
		float[] chw = new float[]{0.4f, 0.6f, 0.8f};
		Mat bgr = EnhanceImageUtils.chwFloatToBgrMat(chw, 1, 1);
		try {
			assertThat(bgr.channels()).isEqualTo(3);
			double[] pixel = bgr.get(0, 0);
			assertThat(pixel[0]).isEqualTo(204d);
			assertThat(pixel[1]).isEqualTo(153d);
			assertThat(pixel[2]).isEqualTo(102d);
		} finally {
			bgr.release();
		}
	}

	@Test
	void clipsOutOfRangeValues() {
		float[] chw = new float[]{-0.2f, 1.5f, 0.5f};
		Mat bgr = EnhanceImageUtils.chwFloatToBgrMat(chw, 1, 1);
		try {
			double[] pixel = bgr.get(0, 0);
			// B 通道来自 chw 第 3 平面（0.5）、G 来自第 2 平面（1.5）、R 来自第 1 平面（-0.2）
			assertThat(pixel[0]).isEqualTo(128d);
			assertThat(pixel[1]).isEqualTo(255d);
			assertThat(pixel[2]).isEqualTo(0d);
		} finally {
			bgr.release();
		}
	}

	@Test
	void rejectsMismatchedChwLength() {
		assertThatThrownBy(() -> EnhanceImageUtils.chwFloatToBgrMat(new float[5], 2, 2))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("长度");
	}

	@Test
	void rejectsNullChw() {
		assertThatThrownBy(() -> EnhanceImageUtils.chwFloatToBgrMat(null, 1, 1))
			.isInstanceOf(MicaAiException.class);
	}

	@Test
	void cropIsIndependentOfSource() {
		Mat src = new Mat(4, 4, CvType.CV_8UC3);
		Mat crop;
		try {
			src.setTo(new org.opencv.core.Scalar(10, 20, 30));
			crop = EnhanceImageUtils.crop(src, 1, 1, 2, 2);
		} finally {
			src.release();
		}
		try {
			assertThat(crop.cols()).isEqualTo(2);
			assertThat(crop.rows()).isEqualTo(2);
			// Scalar(10,20,30) 按 B/G/R 顺序填充，get 返回的数组同样是 B/G/R
			assertThat(crop.get(0, 0)[2]).isEqualTo(30d);
		} finally {
			crop.release();
		}
	}

	@Test
	void encodesPng() {
		Mat bgr = new Mat(2, 2, CvType.CV_8UC3);
		try {
			bgr.setTo(new org.opencv.core.Scalar(1, 2, 3));
			byte[] png = EnhanceImageUtils.imencodePng(bgr);
			assertThat(png).isNotEmpty();
			assertThat(png[0]).isEqualTo((byte) 0x89);
			assertThat(png[1]).isEqualTo((byte) 0x50);
		} finally {
			bgr.release();
		}
	}

	@Test
	void decodesImageBytes() {
		byte[] bytes = loadSubjectImage();
		Mat mat = EnhanceImageUtils.byteArrayToMat(bytes);
		try {
			assertThat(mat.cols()).isEqualTo(480);
			assertThat(mat.rows()).isEqualTo(360);
			assertThat(mat.channels()).isEqualTo(3);
		} finally {
			mat.release();
		}
	}

	@Test
	void rejectsEmptyImageBytes() {
		assertThatThrownBy(() -> EnhanceImageUtils.byteArrayToMat(new byte[0]))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("图像字节为空");
	}

	@Test
	void rejectsInvalidImageBytes() {
		assertThatThrownBy(() -> EnhanceImageUtils.byteArrayToMat(new byte[]{1, 2, 3, 4}))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("解码失败");
	}

	@Test
	void releaseAllIsNullSafe() {
		EnhanceImageUtils.releaseAll((Mat) null, new Mat());
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
}
