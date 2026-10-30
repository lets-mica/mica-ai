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
package net.dreamlu.mica.ai.detect.util;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link DetectImageUtils} 单元测试（letterbox 几何 + CHW 张量化）。
 */
class DetectImageUtilsTest {

	@BeforeAll
	static void loadOpenCv() {
		nu.pattern.OpenCV.loadLocally();
	}

	@AfterAll
	static void tearDown() {
		// 原生库由 JVM 自行回收，无需显式释放
	}

	@Test
	void letterboxSquareImageHasNoPadding() {
		Mat src = new Mat(416, 416, CvType.CV_8UC3, new Scalar(1, 2, 3));
		DetectImageUtils.Letterbox lb = null;
		try {
			lb = DetectImageUtils.letterbox(src, 416, 0, Imgproc.INTER_AREA);
			assertThat(lb.ratio).isEqualTo(1f);
			assertThat(lb.top).isEqualTo(0);
			assertThat(lb.left).isEqualTo(0);
			assertThat(lb.image.cols()).isEqualTo(416);
			assertThat(lb.image.rows()).isEqualTo(416);
		} finally {
			if (lb != null) {
				lb.image.release();
			}
			src.release();
		}
	}

	@Test
	void letterboxWideImageIsCenteredVertically() {
		Mat src = new Mat(400, 800, CvType.CV_8UC3, new Scalar(0, 0, 0));
		DetectImageUtils.Letterbox lb = null;
		try {
			lb = DetectImageUtils.letterbox(src, 416, 0, Imgproc.INTER_AREA);
			assertThat(lb.ratio).isEqualTo(416f / 800f);
			assertThat(lb.left).isEqualTo(0);
			assertThat(lb.top).isEqualTo((416 - 208) / 2);
			assertThat(lb.image.cols()).isEqualTo(416);
			assertThat(lb.image.rows()).isEqualTo(416);
		} finally {
			if (lb != null) {
				lb.image.release();
			}
			src.release();
		}
	}

	@Test
	void letterboxTallImageIsCenteredHorizontally() {
		Mat src = new Mat(800, 400, CvType.CV_8UC3, new Scalar(0, 0, 0));
		DetectImageUtils.Letterbox lb = null;
		try {
			lb = DetectImageUtils.letterbox(src, 416, 0, Imgproc.INTER_AREA);
			assertThat(lb.ratio).isEqualTo(416f / 800f);
			assertThat(lb.top).isEqualTo(0);
			assertThat(lb.left).isEqualTo((416 - 208) / 2);
		} finally {
			if (lb != null) {
				lb.image.release();
			}
			src.release();
		}
	}

	@Test
	void letterboxPadValueAppliedToBorder() {
		Mat src = new Mat(100, 200, CvType.CV_8UC3, new Scalar(255, 255, 255));
		DetectImageUtils.Letterbox lb = null;
		try {
			lb = DetectImageUtils.letterbox(src, 416, 114, Imgproc.INTER_LINEAR);
			double[] topLeft = lb.image.get(0, 0);
			assertThat(topLeft[0]).isEqualTo(114d);
			assertThat(topLeft[1]).isEqualTo(114d);
			assertThat(topLeft[2]).isEqualTo(114d);
		} finally {
			if (lb != null) {
				lb.image.release();
			}
			src.release();
		}
	}

	@Test
	void toChwFloatKeepsRawValuesWhenNotNormalized() {
		Mat rgb = new Mat(1, 1, CvType.CV_8UC3, new Scalar(10, 20, 30));
		try {
			float[] chw = DetectImageUtils.toChwFloat(rgb, null, null);
			assertThat(chw).hasSize(3);
			assertThat(chw[0]).isEqualTo(10f);
			assertThat(chw[1]).isEqualTo(20f);
			assertThat(chw[2]).isEqualTo(30f);
		} finally {
			rgb.release();
		}
	}

	@Test
	void toChwFloatAppliesMeanAndStdPerChannel() {
		Mat rgb = new Mat(1, 1, CvType.CV_8UC3, new Scalar(100, 100, 100));
		try {
			float[] chw = DetectImageUtils.toChwFloat(rgb,
				new float[]{0f, 0f, 0f}, new float[]{2f, 2f, 2f});
			assertThat(chw[0]).isEqualTo(50f);
			assertThat(chw[1]).isEqualTo(50f);
			assertThat(chw[2]).isEqualTo(50f);
		} finally {
			rgb.release();
		}
	}

	@Test
	void toChwFloatUsesChwLayout() {
		Mat rgb = new Mat(1, 2, CvType.CV_8UC3, new Scalar(0, 0, 0));
		rgb.put(0, 0, new byte[]{1, 2, 3, 4, 5, 6});
		try {
			float[] chw = DetectImageUtils.toChwFloat(rgb, null, null);
			assertThat(chw).hasSize(6);
			// 通道优先：R 通道 2 个像素，然后 G，最后 B
			assertThat(chw[0]).isEqualTo(1f);
			assertThat(chw[1]).isEqualTo(4f);
			assertThat(chw[2]).isEqualTo(2f);
			assertThat(chw[3]).isEqualTo(5f);
			assertThat(chw[4]).isEqualTo(3f);
			assertThat(chw[5]).isEqualTo(6f);
		} finally {
			rgb.release();
		}
	}

	@Test
	void byteArrayToMatDecodesPng() throws Exception {
		BufferedImage image = new BufferedImage(8, 6, BufferedImage.TYPE_3BYTE_BGR);
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		ImageIO.write(image, "png", out);
		byte[] bytes = out.toByteArray();

		Mat mat = DetectImageUtils.byteArrayToMat(bytes);
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
		assertThatThrownBy(() -> DetectImageUtils.byteArrayToMat(new byte[0]))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("图像字节为空");
	}

	@Test
	void byteArrayToMatRejectsGarbage() {
		assertThatThrownBy(() -> DetectImageUtils.byteArrayToMat(new byte[]{1, 2, 3, 4}))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("解码失败");
	}

	@Test
	void releaseAllSkipsNull() {
		Mat mat = new Mat();
		DetectImageUtils.releaseAll(null, mat, null);
		assertThat(mat.empty()).isTrue();
	}
}
