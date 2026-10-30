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
package net.dreamlu.mica.ai.humanseg.util;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * {@link HumanSegImageUtils} 单元测试（CHW 张量化 / 掩码缩放 / alpha 合成）。
 */
class HumanSegImageUtilsTest {

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	@Test
	void bgrHwcToRgbChwFloatAppliesChannelOrderAndNormalization() {
		Mat bgr = new Mat(1, 2, CvType.CV_8UC3);
		bgr.put(0, 0, new byte[]{1, 2, 3, 4, 5, 6});
		try {
			// mean=0 / std=1 等价于只做 (x/255)，便于断言数值与通道序
			float[] chw = HumanSegImageUtils.bgrHwcToRgbChwFloat(bgr,
				new float[]{0f, 0f, 0f}, new float[]{1f, 1f, 1f});

			assertThat(chw).hasSize(6);
			// 像素 0 的 BGR 是 (1,2,3) → RGB=(3,2,1)；像素 1 的 BGR 是 (4,5,6) → RGB=(6,5,4)
			assertThat(chw[0]).isCloseTo(3f / 255f, within(1e-6f));
			assertThat(chw[1]).isCloseTo(6f / 255f, within(1e-6f));
			assertThat(chw[2]).isCloseTo(2f / 255f, within(1e-6f));
			assertThat(chw[3]).isCloseTo(5f / 255f, within(1e-6f));
			assertThat(chw[4]).isCloseTo(1f / 255f, within(1e-6f));
			assertThat(chw[5]).isCloseTo(4f / 255f, within(1e-6f));
		} finally {
			bgr.release();
		}
	}

	@Test
	void bgrHwcToRgbChwFloatMapsBlackAndWhiteToMinusOneAndOne() {
		Mat black = new Mat(1, 1, CvType.CV_8UC3, new Scalar(0, 0, 0));
		Mat white = new Mat(1, 1, CvType.CV_8UC3, new Scalar(255, 255, 255));
		try {
			float[] min = HumanSegImageUtils.bgrHwcToRgbChwFloat(black,
				new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f});
			float[] max = HumanSegImageUtils.bgrHwcToRgbChwFloat(white,
				new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f});

			assertThat(min).containsExactly(-1f, -1f, -1f);
			assertThat(max).containsExactly(1f, 1f, 1f);
		} finally {
			black.release();
			white.release();
		}
	}

	@Test
	void resizeToRgbChwFloatProducesChwSizedArray() {
		Mat bgr = new Mat(6, 8, CvType.CV_8UC3, new Scalar(128, 128, 128));
		try {
			float[] chw = HumanSegImageUtils.resizeToRgbChwFloat(bgr, 4, 3,
				new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f});

			assertThat(chw).hasSize(3 * 4 * 3);
			assertThat(chw[0]).isCloseTo(0.0039f, within(1e-3f));
		} finally {
			bgr.release();
		}
	}

	@Test
	void resizeMaskTargetsGivenWidthAndHeight() {
		Mat mask = new Mat(2, 2, CvType.CV_32FC1, new Scalar(0.25));
		Mat resized = null;
		try {
			resized = HumanSegImageUtils.resizeMask(mask, 6, 4);
			assertThat(resized.type()).isEqualTo(CvType.CV_32FC1);
			assertThat(resized.cols()).isEqualTo(6);
			assertThat(resized.rows()).isEqualTo(4);
			assertThat(resized.get(0, 0)[0]).isCloseTo(0.25, within(1e-5));
		} finally {
			mask.release();
			HumanSegImageUtils.releaseAll(resized);
		}
	}

	@Test
	void thresholdBinarizesMaskToZeroOr255() {
		Mat alpha = new Mat(1, 4, CvType.CV_32FC1);
		alpha.put(0, 0, new float[]{0f, 0.5f, 0.6f, 1f});
		Mat binary = null;
		try {
			binary = HumanSegImageUtils.threshold(alpha, 0.5f);
			assertThat(binary.type()).isEqualTo(CvType.CV_8UC1);
			assertThat(binary.get(0, 0)[0]).isEqualTo(0d);
			assertThat(binary.get(0, 1)[0]).isEqualTo(0d);
			assertThat(binary.get(0, 2)[0]).isEqualTo(255d);
			assertThat(binary.get(0, 3)[0]).isEqualTo(255d);
		} finally {
			alpha.release();
			HumanSegImageUtils.releaseAll(binary);
		}
	}

	@Test
	void toBgraWritesAlphaIntoFourthChannel() {
		Mat bgr = new Mat(1, 1, CvType.CV_8UC3, new Scalar(10, 20, 30));
		Mat alpha = new Mat(1, 1, CvType.CV_32FC1, new Scalar(1d));
		Mat bgra = null;
		try {
			bgra = HumanSegImageUtils.toBgra(bgr, alpha);
			assertThat(bgra.channels()).isEqualTo(4);
			double[] px = bgra.get(0, 0);
			assertThat(px[0]).isEqualTo(10d);
			assertThat(px[1]).isEqualTo(20d);
			assertThat(px[2]).isEqualTo(30d);
			assertThat(px[3]).isEqualTo(255d);
		} finally {
			HumanSegImageUtils.releaseAll(bgr, alpha, bgra);
		}
	}

	@Test
	void compositeOnColorRespectsAlpha() {
		Mat fgSource = new Mat(1, 1, CvType.CV_8UC3, new Scalar(10, 20, 30));
		Mat bgSource = new Mat(1, 1, CvType.CV_8UC3, new Scalar(10, 20, 30));
		Mat fgAlpha = new Mat(1, 1, CvType.CV_32FC1, new Scalar(1d));
		Mat bgAlpha = new Mat(1, 1, CvType.CV_32FC1, new Scalar(0d));
		Mat composedFg = null;
		Mat composedBg = null;
		try {
			// 底色 RGB = 纯红 → BGR = (0,0,255)
			composedFg = HumanSegImageUtils.compositeOnColor(fgSource, fgAlpha, new int[]{255, 0, 0});
			composedBg = HumanSegImageUtils.compositeOnColor(bgSource, bgAlpha, new int[]{255, 0, 0});

			double[] fgPx = composedFg.get(0, 0);
			assertThat(fgPx[0]).isEqualTo(10d);
			assertThat(fgPx[1]).isEqualTo(20d);
			assertThat(fgPx[2]).isEqualTo(30d);

			double[] bgPx = composedBg.get(0, 0);
			assertThat(bgPx[0]).isEqualTo(0d);
			assertThat(bgPx[1]).isEqualTo(0d);
			assertThat(bgPx[2]).isEqualTo(255d);
		} finally {
			HumanSegImageUtils.releaseAll(fgSource, bgSource, fgAlpha, bgAlpha,
				composedFg, composedBg);
		}
	}

	@Test
	void byteArrayToMatDecodesPng() throws Exception {
		BufferedImage image = new BufferedImage(8, 6, BufferedImage.TYPE_3BYTE_BGR);
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		ImageIO.write(image, "png", out);

		Mat mat = HumanSegImageUtils.byteArrayToMat(out.toByteArray());
		try {
			assertThat(mat.cols()).isEqualTo(8);
			assertThat(mat.rows()).isEqualTo(6);
			assertThat(mat.channels()).isEqualTo(3);
		} finally {
			mat.release();
		}
	}

	@Test
	void byteArrayToMatRejectsEmptyInput() {
		assertThatThrownBy(() -> HumanSegImageUtils.byteArrayToMat(new byte[0]))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("图像字节为空");
	}

	@Test
	void byteArrayToMatRejectsGarbage() {
		assertThatThrownBy(() -> HumanSegImageUtils.byteArrayToMat(new byte[]{1, 2, 3, 4}))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("解码失败");
	}

	@Test
	void imencodePngProducesDecodablePng() {
		Mat bgra = new Mat(4, 4, CvType.CV_8UC4, new Scalar(1, 2, 3, 4));
		try {
			byte[] png = HumanSegImageUtils.imencodePng(bgra);
			assertThat(png).isNotEmpty();

			Mat decoded = HumanSegImageUtils.byteArrayToMat(png);
			try {
				assertThat(decoded.cols()).isEqualTo(4);
				assertThat(decoded.rows()).isEqualTo(4);
			} finally {
				decoded.release();
			}
		} finally {
			bgra.release();
		}
	}

	@Test
	void releaseAllSkipsNull() {
		Mat mat = new Mat();
		HumanSegImageUtils.releaseAll(null, mat, null);
		assertThat(mat.empty()).isTrue();
	}
}
