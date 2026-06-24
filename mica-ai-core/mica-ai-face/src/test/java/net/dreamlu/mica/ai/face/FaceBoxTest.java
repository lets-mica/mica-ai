/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face;

import net.dreamlu.mica.ai.face.config.FaceBox;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * {@link FaceBox} 几何与尺寸工具方法测试。
 */
class FaceBoxTest {

	@Test
	void width_height_area_basic() {
		FaceBox box = new FaceBox(10f, 20f, 110f, 70f, 0.9f, null);
		assertEquals(100f, box.width(), 1e-6);
		assertEquals(50f, box.height(), 1e-6);
		assertEquals(5000f, box.area(), 1e-6);
	}

	@Test
	void landmarks_passed_through() {
		float[] lmk = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f};
		FaceBox box = new FaceBox(0f, 0f, 100f, 100f, 1f, lmk);
		assertSame(lmk, box.getLandmarks());
	}

	@Test
	void null_landmarks_returns_null_size() {
		FaceBox box = new FaceBox(0f, 0f, 100f, 100f, 1f, null);
		assertNull(box.getLandmarks());
	}
}
