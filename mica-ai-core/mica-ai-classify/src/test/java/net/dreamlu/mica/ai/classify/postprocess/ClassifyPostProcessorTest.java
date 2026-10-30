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
package net.dreamlu.mica.ai.classify.postprocess;

import net.dreamlu.mica.ai.classify.model.ClassifyTopK;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * {@link ClassifyPostProcessor} 单元测试。
 */
class ClassifyPostProcessorTest {

	private static final String[] LABELS = {"0", "90", "180", "270"};

	@Test
	void softmaxNormalizesToSumOne() {
		float[] probs = ClassifyPostProcessor.softmax(new float[]{1f, 2f, 3f});

		assertThat(probs).hasSize(3);
		assertThat(probs[0] + probs[1] + probs[2]).isCloseTo(1f, within(1e-6f));
		assertThat(probs[2]).isGreaterThan(probs[1]);
		assertThat(probs[1]).isGreaterThan(probs[0]);
	}

	@Test
	void softmaxIsNumericallyStableForLargeLogits() {
		float[] probs = ClassifyPostProcessor.softmax(new float[]{1000f, 1001f});

		assertThat(probs[0] + probs[1]).isCloseTo(1f, within(1e-6f));
		assertThat(probs[1]).isGreaterThan(probs[0]);
	}

	@Test
	void softmaxHandlesEmptyInput() {
		assertThat(ClassifyPostProcessor.softmax(null)).isEmpty();
		assertThat(ClassifyPostProcessor.softmax(new float[0])).isEmpty();
	}

	@Test
	void sigmoidMapsEachLogitIndependently() {
		float[] probs = ClassifyPostProcessor.sigmoid(new float[]{0f, 10f, -10f});

		assertThat(probs[0]).isCloseTo(0.5f, within(1e-6f));
		assertThat(probs[1]).isCloseTo(1f, within(1e-4f));
		assertThat(probs[2]).isCloseTo(0f, within(1e-4f));
	}

	@Test
	void topKReturnsDescendingScores() {
		float[] scores = {0.1f, 0.9f, 0.4f, 0.2f};

		List<ClassifyTopK> top = ClassifyPostProcessor.topK(scores, LABELS, 2);

		assertThat(top).hasSize(2);
		assertThat(top.get(0).getIndex()).isEqualTo(1);
		assertThat(top.get(0).getLabel()).isEqualTo("90");
		assertThat(top.get(0).getScore()).isEqualTo(0.9f);
		assertThat(top.get(1).getIndex()).isEqualTo(2);
		assertThat(top.get(1).getLabel()).isEqualTo("180");
	}

	@Test
	void topKClampsToClassCount() {
		float[] scores = {0.1f, 0.9f, 0.4f, 0.2f};

		List<ClassifyTopK> top = ClassifyPostProcessor.topK(scores, LABELS, 10);

		assertThat(top).hasSize(4);
	}

	@Test
	void topKIsStableOnTies() {
		float[] scores = {0.5f, 0.5f, 0.5f, 0.5f};

		List<ClassifyTopK> top = ClassifyPostProcessor.topK(scores, LABELS, 4);

		assertThat(top).extracting(ClassifyTopK::getIndex)
			.containsExactly(0, 1, 2, 3);
	}

	@Test
	void topKFallsBackToIndexWhenLabelsMissing() {
		float[] scores = {0.2f, 0.8f};

		List<ClassifyTopK> top = ClassifyPostProcessor.topK(scores, null, 1);

		assertThat(top).hasSize(1);
		assertThat(top.get(0).getIndex()).isEqualTo(1);
		assertThat(top.get(0).getLabel()).isEqualTo("1");
	}

	@Test
	void topKHandlesEmptyScores() {
		assertThat(ClassifyPostProcessor.topK(null, LABELS, 1)).isEmpty();
		assertThat(ClassifyPostProcessor.topK(new float[0], LABELS, 1)).isEmpty();
	}
}
