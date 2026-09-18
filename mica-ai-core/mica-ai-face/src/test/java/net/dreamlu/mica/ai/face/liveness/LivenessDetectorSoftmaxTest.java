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
package net.dreamlu.mica.ai.face.liveness;

import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * LivenessDetector softmax 单测。
 */
class LivenessDetectorSoftmaxTest {

	private static final Offset<Double> EPS = Offset.offset(1e-6);

	@Test
	void softmaxSumsToOne() {
		float[] logits = {1.0f, 2.0f, 3.0f, 4.0f};
		float[] probs = LivenessDetector.softmax(logits);
		double sum = 0;
		for (float p : probs) {
			sum += p;
		}
		assertThat(sum).isCloseTo(1.0, EPS);
	}

	@Test
	void softmaxArgmaxMatchesLargestLogit() {
		float[] logits = {0.5f, 3.0f, 1.0f};
		float[] probs = LivenessDetector.softmax(logits);
		int argmax = 0;
		float max = probs[0];
		for (int i = 1; i < probs.length; i++) {
			if (probs[i] > max) {
				max = probs[i];
				argmax = i;
			}
		}
		assertThat(argmax).isEqualTo(1);
	}
}