/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face;

import net.dreamlu.mica.ai.face.config.FaceBox;
import net.dreamlu.mica.ai.face.utils.ImageUtils;
import org.junit.jupiter.api.Test;

import java.awt.image.BufferedImage;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * {@link ImageUtils} 图像处理工具测试（letterbox / 对齐 / BGR 转换）。
 */
class ImageUtilsTest {

	@Test
	void letterbox_returns_square() {
		BufferedImage src = new BufferedImage(800, 400, BufferedImage.TYPE_3BYTE_BGR);
		var lb = ImageUtils.letterbox(src, 640);
		assertEquals(640, lb.image().getWidth());
		assertEquals(640, lb.image().getHeight());
		assertEquals(0.8f, lb.scale(), 1e-6);
		assertEquals(640, lb.newW());
		assertEquals(320, lb.newH());
		assertEquals(0, lb.dx());
		assertEquals(160, lb.dy());
	}

	@Test
	void mapBoxesBackToOriginal_undoes_letterbox() {
		BufferedImage src = new BufferedImage(800, 400, BufferedImage.TYPE_3BYTE_BGR);
		var lb = ImageUtils.letterbox(src, 640);
		float[] lmk = new float[10];
		for (int i = 0; i < 10; i++) {
			lmk[i] = 320;
		}
		FaceBox box = new FaceBox(200f, 300f, 400f, 500f, 0.9f, lmk);
		List<FaceBox> mapped = ImageUtils.mapBoxesBackToOriginal(List.of(box), lb);
		assertEquals(1, mapped.size());
		FaceBox m = mapped.get(0);
		assertEquals(250f, m.getX1(), 1e-3);
		assertEquals(175f, m.getY1(), 1e-3);
		assertEquals(500f, m.getX2(), 1e-3);
		assertEquals(425f, m.getY2(), 1e-3);
	}

	@Test
	void toNchwFloat_normalized_rgb_order() {
		BufferedImage img = new BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR);
		img.setRGB(0, 0, 2, 2, new int[]{0x0A141E, 0x0A141E, 0x0A141E, 0x0A141E}, 0, 2);
		float[] data = ImageUtils.toNchwFloat(img, true);
		int planeSize = 4;
		assertEquals(-0.91796875f, data[0], 1e-5);
		assertEquals(-0.83984375f, data[planeSize], 1e-5);
		assertEquals(-0.76171875f, data[2 * planeSize], 1e-5);
	}

	@Test
	void toNchwFloat_unnormalized_keeps_raw_values() {
		BufferedImage img = new BufferedImage(1, 1, BufferedImage.TYPE_3BYTE_BGR);
		img.setRGB(0, 0, 0x0A141E);
		float[] data = ImageUtils.toNchwFloat(img, false);
		assertEquals(10f, data[0], 1e-6);
		assertEquals(20f, data[1], 1e-6);
		assertEquals(30f, data[2], 1e-6);
	}

	@Test
	void align_with_landmarks_produces_112x112() {
		BufferedImage src = new BufferedImage(640, 480, BufferedImage.TYPE_3BYTE_BGR);
		float[] lmk = {200, 200, 300, 200, 250, 300, 220, 400, 280, 400};
		FaceBox box = new FaceBox(150f, 150f, 350f, 450f, 0.9f, lmk);
		BufferedImage aligned = ImageUtils.align(src, box, 112);
		assertNotNull(aligned);
		assertEquals(112, aligned.getWidth());
		assertEquals(112, aligned.getHeight());
		assertEquals(BufferedImage.TYPE_3BYTE_BGR, aligned.getType());
	}

	@Test
	void align_without_landmarks_falls_back_to_crop() {
		BufferedImage src = new BufferedImage(640, 480, BufferedImage.TYPE_3BYTE_BGR);
		FaceBox box = new FaceBox(100f, 100f, 300f, 400f, 0.9f, null);
		BufferedImage aligned = ImageUtils.align(src, box, 112);
		assertEquals(112, aligned.getWidth());
		assertEquals(112, aligned.getHeight());
	}

	@Test
	void align_single_color_image_is_not_all_black() {
		BufferedImage src = new BufferedImage(640, 480, BufferedImage.TYPE_3BYTE_BGR);
		java.awt.Graphics2D g = src.createGraphics();
		try {
			g.setColor(java.awt.Color.RED);
			g.fillRect(0, 0, 640, 480);
		} finally {
			g.dispose();
		}
		float[] lmk = {200, 200, 300, 200, 250, 300, 220, 400, 280, 400};
		FaceBox box = new FaceBox(150f, 150f, 350f, 450f, 0.9f, lmk);
		BufferedImage aligned = ImageUtils.align(src, box, 112);
		boolean foundColoredPixel = false;
		int pixelCount = 0;
		int redCount = 0;
		for (int y = 0; y < 112 && !foundColoredPixel; y++) {
			for (int x = 0; x < 112 && !foundColoredPixel; x++) {
				int rgb = aligned.getRGB(x, y);
				pixelCount++;
				int r = (rgb >> 16) & 0xFF;
				int g_ = (rgb >> 8) & 0xFF;
				int b = rgb & 0xFF;
				if (r > 200 && g_ < 50 && b < 50) {
					redCount++;
					if (redCount > 10) {
						foundColoredPixel = true;
					}
				}
			}
		}
		final int finalPixelCount = pixelCount;
		final int finalRedCount = redCount;
		org.junit.jupiter.api.Assertions.assertTrue(foundColoredPixel,
			() -> "应该至少有 10 个红色像素，扫描了 " + finalPixelCount + " 个像素，找到 " + finalRedCount + " 个红色像素");
	}
}
