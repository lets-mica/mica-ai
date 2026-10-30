/*
 * Copyright (c) 2019-2029, DreamLu 卢春梦 (596392912@qq.com & dreamlu.net).
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
package net.dreamlu.mica.ai.face.util;

import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * {@link ImageUtils#bgrHwcToRgbChwFloat} 的张量布局单测。
 *
 * <p>表情识别与质量评估两个模型都吃 {@code [1, 3, 112, 112]} 的 NCHW 张量，
 * 而 OpenCV 的 {@link Mat} 是 HWC。若内存填充顺序按 HWC 走、却声明成 NCHW，
 * 模型收到的是被打乱的图像 —— 表情识别会退化成恒定输出（什么都识别成「平静」）。
 *
 * <p>本测试用<b>通道值互不相同</b>的像素固定住三个维度：
 * 通道顺序（R/G/B）、平面顺序（CHW 而非 HWC）、归一化公式。
 */
class ImageUtilsChwTest {

	static {
		nu.pattern.OpenCV.loadLocally();
	}

	/**
	 * 构造 2×2 的 BGR 测试图，四个像素的三通道值各不相同。
	 *
	 * @return BGR Mat（{@code CV_8UC3}）
	 */
	private static Mat buildDistinctBgr() {
		Mat mat = new Mat(2, 2, CvType.CV_8UC3);
		mat.put(0, 0, new double[]{10, 20, 30});
		mat.put(0, 1, new double[]{40, 50, 60});
		mat.put(1, 0, new double[]{70, 80, 90});
		mat.put(1, 1, new double[]{100, 110, 120});
		return mat;
	}

	@Test
	void layoutIsChannelFirstWithRgbOrder() {
		Mat bgr = buildDistinctBgr();
		try {
			// 三通道的 mean / std 各不相同，通道反序会被立刻发现
			float[] mean = {0.1f, 0.2f, 0.3f};
			float[] std = {0.5f, 0.6f, 0.7f};

			float[] data = ImageUtils.bgrHwcToRgbChwFloat(bgr, mean, std);

			assertThat(data).hasSize(3 * 2 * 2);

			// 前 4 个是 R 平面（CHW 的 C=0），按行序排列
			assertThat(data[0]).isCloseTo((30 / 255f - 0.1f) / 0.5f, within(1e-6f));
			assertThat(data[1]).isCloseTo((60 / 255f - 0.1f) / 0.5f, within(1e-6f));
			assertThat(data[2]).isCloseTo((90 / 255f - 0.1f) / 0.5f, within(1e-6f));
			assertThat(data[3]).isCloseTo((120 / 255f - 0.1f) / 0.5f, within(1e-6f));

			// 接着 4 个是 G 平面（C=1）
			assertThat(data[4]).isCloseTo((20 / 255f - 0.2f) / 0.6f, within(1e-6f));
			assertThat(data[5]).isCloseTo((50 / 255f - 0.2f) / 0.6f, within(1e-6f));
			assertThat(data[6]).isCloseTo((80 / 255f - 0.2f) / 0.6f, within(1e-6f));
			assertThat(data[7]).isCloseTo((110 / 255f - 0.2f) / 0.6f, within(1e-6f));

			// 最后 4 个是 B 平面（C=2）
			assertThat(data[8]).isCloseTo((10 / 255f - 0.3f) / 0.7f, within(1e-6f));
			assertThat(data[9]).isCloseTo((40 / 255f - 0.3f) / 0.7f, within(1e-6f));
			assertThat(data[10]).isCloseTo((70 / 255f - 0.3f) / 0.7f, within(1e-6f));
			assertThat(data[11]).isCloseTo((100 / 255f - 0.3f) / 0.7f, within(1e-6f));
		} finally {
			bgr.release();
		}
	}

	@Test
	void differsFromHwcLayout() {
		Mat bgr = buildDistinctBgr();
		try {
			float[] mean = {0.1f, 0.2f, 0.3f};
			float[] std = {0.5f, 0.6f, 0.7f};

			float[] chw = ImageUtils.bgrHwcToRgbChwFloat(bgr, mean, std);

			// HWC 布局下，前 3 个元素是「像素0 的 R / G / B」
			float hwcR0 = (30 / 255f - 0.1f) / 0.5f;
			float hwcG0 = (20 / 255f - 0.2f) / 0.6f;
			float hwcB0 = (10 / 255f - 0.3f) / 0.7f;

			// CHW 布局下，首元素同样是像素0 的 R，从第 2 个元素起就分叉
			assertThat(chw[0]).isEqualTo(hwcR0);
			assertThat(chw[1]).isNotEqualTo(hwcG0);
			assertThat(chw[2]).isNotEqualTo(hwcB0);
		} finally {
			bgr.release();
		}
	}
}
