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
package net.dreamlu.mica.ai.track.motion;

import net.dreamlu.mica.ai.track.model.BBox;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

/**
 * {@link AffineMotionModel} 仿射变换测试（纯几何，零 ONNX / OpenCV 依赖）。
 */
class AffineMotionModelTest {

	private static BBox box() {
		return new BBox(100f, 100f, 140f, 140f, 0.9f, 0);
	}

	@Test
	void identityKeepsBoxUnchanged() {
		AffineMotionModel model = AffineMotionModel.identity();

		BBox result = model.toCurrent(box());

		assertThat(model.isIdentity()).isTrue();
		assertThat(result.getX1()).isEqualTo(100f);
		assertThat(result.getY1()).isEqualTo(100f);
		assertThat(result.getX2()).isEqualTo(140f);
		assertThat(result.getY2()).isEqualTo(140f);
	}

	@Test
	void translationShiftsBox() {
		AffineMotionModel model = AffineMotionModel.translation(12f, -8f);

		BBox result = model.toCurrent(box());

		assertThat(result.getX1()).isEqualTo(112f);
		assertThat(result.getY1()).isEqualTo(92f);
		assertThat(result.getX2()).isEqualTo(152f);
		assertThat(result.getY2()).isEqualTo(132f);
	}

	@Test
	void rotationProducesAxisAlignedBoundingBox() {
		// 相机滚转 = 绕主点（框中心）旋转
		AffineMotionModel model = AffineMotionModel.rotationAround((float) (Math.PI / 4d), 120f, 120f);

		BBox result = model.toCurrent(box());

		// 旋转 45° 后轴对齐外接矩形必然变大 —— 这是几何事实
		assertThat(result.width()).isGreaterThan(40f);
		assertThat(result.height()).isGreaterThan(40f);
		// 绕框中心旋转，中心点保持不变
		assertThat(result.cx()).isCloseTo(120f, within(0.01f));
		assertThat(result.cy()).isCloseTo(120f, within(0.01f));
	}

	@Test
	void rotationAroundOriginMovesBoxAway() {
		AffineMotionModel model = AffineMotionModel.rotation((float) (Math.PI / 2d));

		BBox result = model.toCurrent(box());

		// 绕原点转 90°：(100,100) → (-100,100)，(140,140) → (-140,140)
		assertThat(result.getX1()).isCloseTo(-140f, within(0.01f));
		assertThat(result.getY1()).isCloseTo(100f, within(0.01f));
		assertThat(result.getX2()).isCloseTo(-100f, within(0.01f));
		assertThat(result.getY2()).isCloseTo(140f, within(0.01f));
	}

	@Test
	void toReferenceIsInverseOfToCurrent() {
		// 轴对齐仿射（缩放 + 平移）：外接矩形不丢信息，逆变换精确还原
		AffineMotionModel model = AffineMotionModel.of(0.98f, 0f, 7f, 0f, 0.99f, -4f);

		BBox current = model.toCurrent(box());
		BBox back = model.toReference(current);

		assertThat(back.getX1()).isCloseTo(100f, within(0.01f));
		assertThat(back.getY1()).isCloseTo(100f, within(0.01f));
		assertThat(back.getX2()).isCloseTo(140f, within(0.01f));
		assertThat(back.getY2()).isCloseTo(140f, within(0.01f));
	}

	@Test
	void toReferenceIsApproximateForRotatedBox() {
		AffineMotionModel model = AffineMotionModel.rotationAround((float) (Math.PI / 36d), 120f, 120f);

		BBox current = model.toCurrent(box());
		BBox back = model.toReference(current);

		// 旋转让轴对齐外接矩形变大，这部分信息在变换到外接矩形时已经丢失 →
		// 逆变换只能还原到「原框被外接矩形包住」的程度
		assertThat(back.getX1()).isLessThanOrEqualTo(100f);
		assertThat(back.getY1()).isLessThanOrEqualTo(100f);
		assertThat(back.getX2()).isGreaterThanOrEqualTo(140f);
		assertThat(back.getY2()).isGreaterThanOrEqualTo(140f);
	}

	@Test
	void appendAccumulatesIncrementalMotion() {
		AffineMotionModel model = AffineMotionModel.identity();

		model.append(AffineMotionModel.translation(-12f, 0f));
		model.append(AffineMotionModel.translation(-12f, 0f));
		model.append(AffineMotionModel.translation(-12f, 0f));

		BBox result = model.toCurrent(box());

		// 三帧累积：参考帧 x=100 → 当前帧 x=100-36
		assertThat(result.getX1()).isCloseTo(64f, within(0.01f));
		assertThat(result.getY1()).isCloseTo(100f, within(0.01f));
	}

	@Test
	void composeAppliesSecondThenFirst() {
		AffineMotionModel first = AffineMotionModel.translation(10f, 0f);
		AffineMotionModel second = AffineMotionModel.translation(5f, 0f);

		AffineMotionModel composed = AffineMotionModel.compose(first, second);

		BBox result = composed.toCurrent(box());
		assertThat(result.getX1()).isCloseTo(115f, within(0.01f));
	}

	@Test
	void resetRestoresIdentity() {
		AffineMotionModel model = AffineMotionModel.translation(30f, 30f);

		model.reset();

		assertThat(model.isIdentity()).isTrue();
	}

	@Test
	void copyIsIndependentFromOriginal() {
		AffineMotionModel model = AffineMotionModel.translation(30f, 30f);

		AffineMotionModel copy = model.copy();
		model.reset();

		assertThat(copy.isIdentity()).isFalse();
		assertThat(copy.toCurrent(box()).getX1()).isCloseTo(130f, within(0.01f));
	}

	@Test
	void degenerateTransformFallsBackToIdentityOnInverse() {
		AffineMotionModel model = AffineMotionModel.of(0f, 0f, 0f, 0f, 0f, 0f);
		BBox input = box();

		BBox result = model.toReference(input);

		// 退化矩阵（缩放为 0）不可逆，退化为恒等
		assertThat(result.getX1()).isEqualTo(100f);
		assertThat(result.getY1()).isEqualTo(100f);
		assertThat(result.getX2()).isEqualTo(140f);
		assertThat(result.getY2()).isEqualTo(140f);
	}

	@Test
	void nullBoxIsTolerated() {
		AffineMotionModel model = AffineMotionModel.translation(5f, 5f);

		assertThat(model.toCurrent(null)).isNull();
		assertThat(model.toReference(null)).isNull();
	}

}
