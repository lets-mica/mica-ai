/*
 * Copyright (c) 2024-2026 mica-ai
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