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
package net.dreamlu.mica.ai.docalign.util;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.docalign.model.DocAlignResult;
import net.dreamlu.mica.ai.docalign.model.DocCorner;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Arrays;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

/**
 * {@link DocAlignImageUtils} 单元测试（CHW 张量化 / 热力图取质心 / 透视矫正）。
 */
class DocAlignImageUtilsTest {

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	@Test
	void bgrHwcToRgbChwFloatAppliesChannelOrderAndDivideBy255() {
		Mat bgr = new Mat(1, 2, CvType.CV_8UC3);
		bgr.put(0, 0, new byte[]{1, 2, 3, 4, 5, 6});
		try {
			float[] chw = DocAlignImageUtils.bgrHwcToRgbChwFloat(bgr);

			assertThat(chw).hasSize(6);
			// 像素 0 的 BGR 是 (1,2,3) → RGB=(3,2,1)
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
	void resizeToRgbChwFloatProducesChwSizedArray() {
		Mat bgr = new Mat(6, 8, CvType.CV_8UC3, new Scalar(255, 255, 255));
		try {
			float[] chw = DocAlignImageUtils.resizeToRgbChwFloat(bgr, 4, 3);

			assertThat(chw).hasSize(3 * 4 * 3);
			for (float v : chw) {
				assertThat(v).isCloseTo(1f, within(1e-5f));
			}
		} finally {
			bgr.release();
		}
	}

	@Test
	void locateCornerReturnsCentroidAndMaxScore() {
		Mat heatmap = new Mat(20, 20, CvType.CV_32FC1, new Scalar(0d));
		try {
			for (int y = 5; y <= 8; y++) {
				for (int x = 5; x <= 8; x++) {
					heatmap.put(y, x, new float[]{0.9f});
				}
			}
			double[] located = DocAlignImageUtils.locateCorner(heatmap, 0.3f);

			assertThat(located[0]).isCloseTo(6.5, within(0.6));
			assertThat(located[1]).isCloseTo(6.5, within(0.6));
			assertThat(located[2]).isCloseTo(0.9, within(1e-5));
		} finally {
			heatmap.release();
		}
	}

	@Test
	void locateCornerPicksLargestBlob() {
		Mat heatmap = new Mat(40, 40, CvType.CV_32FC1, new Scalar(0d));
		try {
			// 小斑点（score 更高）与大连通域（面积更大），上游语义取面积最大者
			for (int y = 2; y <= 3; y++) {
				for (int x = 2; x <= 3; x++) {
					heatmap.put(y, x, new float[]{0.95f});
				}
			}
			for (int y = 30; y <= 35; y++) {
				for (int x = 28; x <= 35; x++) {
					heatmap.put(y, x, new float[]{0.6f});
				}
			}
			double[] located = DocAlignImageUtils.locateCorner(heatmap, 0.3f);

			assertThat(located[0]).isCloseTo(31.5, within(1.0));
			assertThat(located[1]).isCloseTo(32.5, within(1.0));
			assertThat(located[2]).isCloseTo(0.95, within(1e-5));
		} finally {
			heatmap.release();
		}
	}

	@Test
	void locateCornerFallsBackToMaxWhenBelowThreshold() {
		Mat heatmap = new Mat(20, 20, CvType.CV_32FC1, new Scalar(0d));
		try {
			heatmap.put(7, 12, new float[]{0.1f});
			double[] located = DocAlignImageUtils.locateCorner(heatmap, 0.3f);

			assertThat(located[0]).isEqualTo(12d);
			assertThat(located[1]).isEqualTo(7d);
			assertThat(located[2]).isCloseTo(0.1, within(1e-6));
		} finally {
			heatmap.release();
		}
	}

	@Test
	void suggestOutputSizeUsesLongestEdges() {
		DocAlignResult result = new DocAlignResult(Arrays.asList(
			new DocCorner(100d, 50d, 0.9),
			new DocCorner(500d, 50d, 0.9),
			new DocCorner(500d, 450d, 0.9),
			new DocCorner(100d, 450d, 0.9)), 600, 500);

		int[] size = DocAlignImageUtils.suggestOutputSize(result);

		assertThat(size[0]).isEqualTo(400);
		assertThat(size[1]).isEqualTo(400);
	}

	@Test
	void warpMapsCornersToFullRectangle() {
		Mat bgr = new Mat(100, 100, CvType.CV_8UC3, new Scalar(10, 20, 30));
		Mat warped = null;
		try {
			DocAlignResult result = new DocAlignResult(Arrays.asList(
				new DocCorner(0d, 0d, 1d),
				new DocCorner(100d, 0d, 1d),
				new DocCorner(100d, 100d, 1d),
				new DocCorner(0d, 100d, 1d)), 100, 100);

			warped = DocAlignImageUtils.warp(bgr, result, 100, 100);
			assertThat(warped.cols()).isEqualTo(100);
			assertThat(warped.rows()).isEqualTo(100);
			assertThat(warped.get(50, 50)[0]).isCloseTo(10d, within(2d));
			assertThat(warped.get(50, 50)[1]).isCloseTo(20d, within(2d));
			assertThat(warped.get(50, 50)[2]).isCloseTo(30d, within(2d));
		} finally {
			bgr.release();
			DocAlignImageUtils.releaseAll(warped);
		}
	}

	@Test
	void warpRejectsNonPositiveOutputSize() {
		DocAlignResult result = new DocAlignResult(Arrays.asList(
			new DocCorner(0d, 0d, 1d),
			new DocCorner(10d, 0d, 1d),
			new DocCorner(10d, 10d, 1d),
			new DocCorner(0d, 10d, 1d)), 10, 10);
		Mat bgr = new Mat(10, 10, CvType.CV_8UC3, new Scalar(0, 0, 0));
		try {
			assertThatThrownBy(() -> DocAlignImageUtils.warp(bgr, result, 0, 10))
				.isInstanceOf(MicaAiException.class)
				.hasMessageContaining("输出宽高必须为正数");
		} finally {
			bgr.release();
		}
	}

	@Test
	void byteArrayToMatDecodesPng() throws Exception {
		BufferedImage image = new BufferedImage(8, 6, BufferedImage.TYPE_3BYTE_BGR);
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		ImageIO.write(image, "png", out);

		Mat mat = DocAlignImageUtils.byteArrayToMat(out.toByteArray());
		try {
			assertThat(mat.cols()).isEqualTo(8);
			assertThat(mat.rows()).isEqualTo(6);
			assertThat(mat.channels()).isEqualTo(3);
		} finally {
			mat.release();
		}
	}

	@Test
	void byteArrayToMatRejectsEmptyAndGarbage() {
		assertThatThrownBy(() -> DocAlignImageUtils.byteArrayToMat(new byte[0]))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("图像字节为空");
		assertThatThrownBy(() -> DocAlignImageUtils.byteArrayToMat(new byte[]{1, 2, 3, 4}))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("解码失败");
	}

	@Test
	void imencodePngProducesDecodablePng() {
		Mat bgr = new Mat(4, 4, CvType.CV_8UC3, new Scalar(1, 2, 3));
		try {
			byte[] png = DocAlignImageUtils.imencodePng(bgr);
			assertThat(png).isNotEmpty();

			Mat decoded = DocAlignImageUtils.byteArrayToMat(png);
			try {
				assertThat(decoded.cols()).isEqualTo(4);
				assertThat(decoded.rows()).isEqualTo(4);
			} finally {
				decoded.release();
			}
		} finally {
			bgr.release();
		}
	}

	@Test
	void releaseAllSkipsNull() {
		Mat mat = new Mat();
		DocAlignImageUtils.releaseAll(null, mat, null);
		assertThat(mat.empty()).isTrue();
	}

	@Test
	void resizeHeatmapTargetsGivenSize() {
		Mat heatmap = new Mat(2, 2, CvType.CV_32FC1, new Scalar(0.5));
		Mat resized = null;
		try {
			resized = DocAlignImageUtils.resizeHeatmap(heatmap, 8, 6);
			assertThat(resized.type()).isEqualTo(CvType.CV_32FC1);
			assertThat(resized.cols()).isEqualTo(8);
			assertThat(resized.rows()).isEqualTo(6);
			assertThat(Core.mean(resized).val[0]).isCloseTo(0.5, within(1e-4));
		} finally {
			heatmap.release();
			DocAlignImageUtils.releaseAll(resized);
		}
	}
}
