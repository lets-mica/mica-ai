/*
 * Copyright (c) 2024-2026 mica-ai
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
}