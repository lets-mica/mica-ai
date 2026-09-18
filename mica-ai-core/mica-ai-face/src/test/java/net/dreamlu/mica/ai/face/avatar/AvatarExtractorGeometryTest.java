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
package net.dreamlu.mica.ai.face.avatar;

import net.dreamlu.mica.ai.face.avatar.AvatarExtractor.Window;
import net.dreamlu.mica.ai.face.model.FaceBox;
import org.assertj.core.data.Offset;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * AvatarExtractor 几何单测 —— 钉死仿射矩阵推导。
 */
class AvatarExtractorGeometryTest {

	private static final Offset<Float> EPS_F = Offset.offset(0.01f);

	@Test
	void windowAffineProducesReasonableSide() {
		FaceBox box = new FaceBox(100f, 100f, 200f, 220f, 0.9f, null);
		AvatarOptions opts = AvatarOptions.defaults();
		Window w = AvatarExtractor.windowAffine(box, opts);
		// 长边 120 * 1.6 = 192 → round = 192
		assertThat(w.side()).isEqualTo(192);
		// 仿射矩阵 6 个数；行列式应非零
		assertThat(w.affine()).hasSize(6);
		double det = w.affine()[0] * w.affine()[4] - w.affine()[1] * w.affine()[3];
		assertThat(Math.abs(det)).isGreaterThan(1e-12);
	}

	@Test
	void rotateForwardAndBackAreInverse() {
		int w = 1000, h = 800;
		float[] p = {123.4f, 567.8f};
		for (int deg : new int[]{0, 90, 180, 270}) {
			float[] fwd = AvatarExtractor.rotateForward(p[0], p[1], deg, w, h);
			float[] back = AvatarExtractor.rotateBack(fwd[0], fwd[1], deg, w, h);
			assertThat(back[0]).isCloseTo(p[0], EPS_F);
			assertThat(back[1]).isCloseTo(p[1], EPS_F);
		}
	}

	/**
	 * 窗口边长只能由「检测框长边 × faceScale」决定，与自动摆正走了哪个基数角无关。
	 *
	 * <p>回归背景：若把摆正后重检出来的框当作窗口基准，同一张脸的头像大小会随朝向漂移
	 * （实测横躺身份证 1.6 系数下边长 132 → 156，多出来的部分会框到卡面文字）。
	 */
	@Test
	void windowSideIsIndependentOfCardinalRotation() {
		FaceBox box = new FaceBox(100f, 100f, 200f, 220f, 0.9f, null);
		AvatarOptions opts = AvatarOptions.defaults();
		int side = AvatarExtractor.windowAffine(box, opts).side();
		assertThat(side).isEqualTo(192);
		for (int deg : new int[]{90, 180, 270}) {
			FaceBox rotated = AvatarExtractor.rotateBox(box, deg, 1000, 800);
			assertThat(AvatarExtractor.windowAffine(rotated, opts).side())
				.as("rot %d", deg)
				.isEqualTo(side);
		}
	}
}