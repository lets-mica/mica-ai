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
package net.dreamlu.mica.ai.classify.util;

import net.dreamlu.mica.ai.classify.config.ClassifyChannelOrder;
import net.dreamlu.mica.ai.classify.config.ClassifyInterpolation;
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
 * {@link ClassifyImageUtils} 单元测试（通道序 / 归一化 / 预处理尺寸 / 解码）。
 */
class ClassifyImageUtilsTest {

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	@Test
	void rgbChannelOrderFlipsOpenCvBgr() {
		Mat bgr = new Mat(1, 2, CvType.CV_8UC3);
		bgr.put(0, 0, new byte[]{1, 2, 3, 4, 5, 6});
		try {
			// mean=0 / std=1 等价于只做 (x/255)，便于断言数值与通道序
			float[] chw = ClassifyImageUtils.hwcToChwFloat(bgr,
				new float[]{0f, 0f, 0f}, new float[]{1f, 1f, 1f},
				ClassifyChannelOrder.RGB);

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
	void bgrChannelOrderKeepsOpenCvOrder() {
		Mat bgr = new Mat(1, 1, CvType.CV_8UC3);
		bgr.put(0, 0, new byte[]{1, 2, 3});
		try {
			float[] chw = ClassifyImageUtils.hwcToChwFloat(bgr,
				new float[]{0f, 0f, 0f}, new float[]{1f, 1f, 1f},
				ClassifyChannelOrder.BGR);

			assertThat(chw[0]).isCloseTo(1f / 255f, within(1e-6f));
			assertThat(chw[1]).isCloseTo(2f / 255f, within(1e-6f));
			assertThat(chw[2]).isCloseTo(3f / 255f, within(1e-6f));
		} finally {
			bgr.release();
		}
	}

	@Test
	void imageNetNormalizationMapsWhiteToPositiveOne() {
		Mat white = new Mat(1, 1, CvType.CV_8UC3, new Scalar(255, 255, 255));
		try {
			float[] chw = ClassifyImageUtils.hwcToChwFloat(white,
				new float[]{0.485f, 0.456f, 0.406f},
				new float[]{0.229f, 0.224f, 0.225f},
				ClassifyChannelOrder.RGB);

			assertThat(chw[0]).isCloseTo((1f - 0.485f) / 0.229f, within(1e-5f));
			assertThat(chw[1]).isCloseTo((1f - 0.456f) / 0.224f, within(1e-5f));
			assertThat(chw[2]).isCloseTo((1f - 0.406f) / 0.225f, within(1e-5f));
		} finally {
			white.release();
		}
	}

	@Test
	void directResizeProducesChwSizedArrayUsingWidthHeightOrder() {
		Mat bgr = new Mat(6, 8, CvType.CV_8UC3, new Scalar(128, 128, 128));
		try {
			float[] chw = ClassifyImageUtils.toChwFloat(bgr, 4, 3, 0,
				new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f},
				ClassifyChannelOrder.RGB, ClassifyInterpolation.LINEAR);

			assertThat(chw).hasSize(3 * 4 * 3);
			assertThat(chw[0]).isCloseTo(0.0039f, within(1e-3f));
		} finally {
			bgr.release();
		}
	}

	@Test
	void shortSideResizeThenCenterCropProducesTargetSize() {
		Mat bgr = new Mat(100, 100, CvType.CV_8UC3, new Scalar(200, 100, 50));
		try {
			float[] chw = ClassifyImageUtils.toChwFloat(bgr, 64, 64, 96,
				new float[]{0f, 0f, 0f}, new float[]{1f, 1f, 1f},
				ClassifyChannelOrder.RGB, ClassifyInterpolation.LINEAR);

			assertThat(chw).hasSize(3 * 64 * 64);
			// 纯色图像中心裁剪后仍为原色：BGR(200,100,50) → RGB=(50,100,200)
			assertThat(chw[0]).isCloseTo(50f / 255f, within(1e-3f));
			assertThat(chw[64 * 64]).isCloseTo(100f / 255f, within(1e-3f));
			assertThat(chw[2 * 64 * 64]).isCloseTo(200f / 255f, within(1e-3f));
		} finally {
			bgr.release();
		}
	}

	@Test
	void byteArrayToMatDecodesPng() throws Exception {
		BufferedImage image = new BufferedImage(8, 6, BufferedImage.TYPE_3BYTE_BGR);
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		ImageIO.write(image, "png", out);

		Mat mat = ClassifyImageUtils.byteArrayToMat(out.toByteArray());
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
		assertThatThrownBy(() -> ClassifyImageUtils.byteArrayToMat(new byte[0]))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("图像字节为空");
	}

	@Test
	void byteArrayToMatRejectsGarbage() {
		assertThatThrownBy(() -> ClassifyImageUtils.byteArrayToMat(new byte[]{1, 2, 3, 4}))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("解码失败");
	}

	@Test
	void releaseAllSkipsNull() {
		Mat mat = new Mat();
		ClassifyImageUtils.releaseAll(null, mat, null);
		assertThat(mat.empty()).isTrue();
	}

	@Test
	void nullInterpolationFallsBackToLinear() {
		Mat bgr = new Mat(4, 4, CvType.CV_8UC3, new Scalar(128, 128, 128));
		try {
			float[] chw = ClassifyImageUtils.toChwFloat(bgr, 2, 2, 0,
				new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f},
				ClassifyChannelOrder.RGB, null);

			assertThat(chw).hasSize(3 * 2 * 2);
		} finally {
			bgr.release();
		}
	}
}
