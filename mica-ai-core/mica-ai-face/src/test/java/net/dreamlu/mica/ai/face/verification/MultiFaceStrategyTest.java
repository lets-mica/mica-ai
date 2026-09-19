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
package net.dreamlu.mica.ai.face.verification;

import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.config.MultiFaceStrategy;
import net.dreamlu.mica.ai.face.model.FaceBox;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link FaceVerifier#selectFace} 选脸策略单测。
 *
 * <p>纯逻辑用例：不加载 ONNX 模型、不依赖 OpenCV，可在任意环境跑。
 */
class MultiFaceStrategyTest {

	/** 面积 100（10x10）、置信度 0.99 —— 面积最小但得分最高。 */
	private static final FaceBox SMALL_HIGH_SCORE = box(0f, 0f, 10f, 10f, 0.99f);
	/** 面积 10000（100x100）、置信度 0.60 —— 面积最大但得分最低。 */
	private static final FaceBox LARGE_LOW_SCORE = box(0f, 0f, 100f, 100f, 0.60f);
	/** 面积 900（30x30）、置信度 0.80 —— 两项都居中。 */
	private static final FaceBox MEDIUM = box(200f, 200f, 230f, 230f, 0.80f);

	private static FaceBox box(float x1, float y1, float x2, float y2, float score) {
		return new FaceBox(x1, y1, x2, y2, score, null);
	}

	@Test
	void defaultStrategyIsLargestArea() {
		assertThat(MultiFaceStrategy.DEFAULT).isEqualTo(MultiFaceStrategy.LARGEST_AREA);
	}

	@Test
	void nullOrEmptyInputGivesNull() {
		assertThat(FaceVerifier.selectFace(null, MultiFaceStrategy.LARGEST_AREA)).isNull();
		assertThat(FaceVerifier.selectFace(Collections.<FaceBox>emptyList(), MultiFaceStrategy.LARGEST_AREA)).isNull();
		// 空列表在 REJECT 策略下也不应抛异常，语义仍是「没有人脸」
		assertThat(FaceVerifier.selectFace(Collections.<FaceBox>emptyList(), MultiFaceStrategy.REJECT)).isNull();
	}

	@Test
	void singleFaceIsReturnedForEveryStrategy() {
		List<FaceBox> one = Collections.singletonList(MEDIUM);
		for (MultiFaceStrategy s : MultiFaceStrategy.values()) {
			assertThat(FaceVerifier.selectFace(one, s)).as("strategy=%s", s).isSameAs(MEDIUM);
		}
	}

	@Test
	void largestAreaWinsEvenWhenItsScoreIsLowest() {
		List<FaceBox> boxes = Arrays.asList(SMALL_HIGH_SCORE, LARGE_LOW_SCORE, MEDIUM);
		assertThat(FaceVerifier.selectFace(boxes, MultiFaceStrategy.LARGEST_AREA)).isSameAs(LARGE_LOW_SCORE);
	}

	@Test
	void largestScoreWinsEvenWhenItsAreaIsSmallest() {
		List<FaceBox> boxes = Arrays.asList(SMALL_HIGH_SCORE, LARGE_LOW_SCORE, MEDIUM);
		assertThat(FaceVerifier.selectFace(boxes, MultiFaceStrategy.LARGEST_SCORE)).isSameAs(SMALL_HIGH_SCORE);
	}

	@Test
	void rejectThrowsOnMultipleFaces() {
		List<FaceBox> boxes = Arrays.asList(SMALL_HIGH_SCORE, LARGE_LOW_SCORE);
		assertThatThrownBy(() -> FaceVerifier.selectFace(boxes, MultiFaceStrategy.REJECT))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("2 张人脸")
			.satisfies(e -> assertThat(((MicaAiException) e).getCode()).isEqualTo(ErrorCode.VERIFICATION_FAILED));
	}

	@Test
	void tiesKeepTheFirstDetectedBox() {
		FaceBox a = box(0f, 0f, 10f, 10f, 0.9f);
		FaceBox b = box(50f, 50f, 60f, 60f, 0.9f);
		List<FaceBox> boxes = Arrays.asList(a, b);
		// 严格大于才替换 ⇒ 并列时保留先检出者（即 detector 的 score 降序序首）
		assertThat(FaceVerifier.selectFace(boxes, MultiFaceStrategy.LARGEST_SCORE)).isSameAs(a);
		assertThat(FaceVerifier.selectFace(boxes, MultiFaceStrategy.LARGEST_AREA)).isSameAs(a);
	}
}
