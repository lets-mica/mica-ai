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
package net.dreamlu.mica.ai.face.recognition;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * FeatureExtractor L2 归一化单测。
 *
 * <p>不依赖 ONNX 原生库：只测 {@link FeatureExtractor#l2NormalizeCopy(float[])}
 * 与 {@link FeatureExtractor#compare(float[], float[])} 的纯算法。
 */
class FeatureExtractorL2Test {

	private static final Offset<Double> EPS = Offset.offset(1e-6);
	private static final Offset<Float> EPS_F = Offset.offset(1e-6f);

	@Test
	void l2NormalizeProducesUnitVector() {
		float[] v = {3f, 4f};
		float[] out = FeatureExtractor.l2NormalizeCopy(v);
		double sum = 0;
		for (float f : out) {
			sum += f * f;
		}
		assertThat(Math.sqrt(sum)).isCloseTo(1.0, EPS);
	}

	@Test
	void compareReturnsDotProductForUnitVectors() {
		float[] a = FeatureExtractor.l2NormalizeCopy(new float[]{1f, 0f, 0f});
		float[] b = FeatureExtractor.l2NormalizeCopy(new float[]{0f, 1f, 0f});
		// 正交向量余弦相似度 = 0
		assertThat(FeatureExtractor.compare(a, b)).isCloseTo(0f, EPS_F);
	}
}