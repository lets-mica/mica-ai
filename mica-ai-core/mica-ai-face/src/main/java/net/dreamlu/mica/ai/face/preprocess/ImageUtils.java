/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.preprocess;

import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.config.FaceBox;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

/**
 * 图像工具：读取 / 缩放 / Letterbox / BGR 转换。
 *
 * <p>零外部依赖，只用 JDK 自带的 {@link BufferedImage}，避免引入 OpenCV。
 *
 * @since 1.0.0
 */
@UtilityClass
public class ImageUtils {

	/**
	 * 从路径读取图像，自动识别 JPEG/PNG/GIF/BMP。
	 */
	public static BufferedImage read(Path path) throws IOException {
		try (InputStream in = Files.newInputStream(path)) {
			return ImageIO.read(in);
		}
	}

	/**
	 * Letterbox 缩放：保持长宽比，短的边补灰边，结果边长 = {@code size}。
	 *
	 * <p>返回 newImage、缩放比例 scale、原图相对 newImage 的左上角偏移 (dx, dy)。
	 */
	public static LetterboxResult letterbox(BufferedImage src, int size) {
		int sw = src.getWidth();
		int sh = src.getHeight();
		float scale = Math.min((float) size / sw, (float) size / sh);
		int nw = Math.round(sw * scale);
		int nh = Math.round(sh * scale);
		int dx = (size - nw) / 2;
		int dy = (size - nh) / 2;

		BufferedImage dst = new BufferedImage(size, size, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = dst.createGraphics();
		try {
			g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
			g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
			g.setColor(new java.awt.Color(114, 114, 114));  // 灰色 padding
			g.fillRect(0, 0, size, size);
			g.drawImage(src, dx, dy, nw, nh, null);
		} finally {
			g.dispose();
		}
		return new LetterboxResult(dst, scale, dx, dy, nw, nh);
	}

	/**
	 * 把 {@link BufferedImage} 转成 ONNX 输入需要的 float32 BCHW 张量。
	 *
	 * <p>像素归一化：减均值 (127.5, 127.5, 127.5) 再除以 128.0，与 InsightFace 原版对齐。
	 */
	public static float[] toNchwFloat(BufferedImage img) {
		int w = img.getWidth();
		int h = img.getHeight();
		byte[] raw = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
		// TYPE_3BYTE_BGR：每像素 3 字节，顺序 B, G, R
		float[] out = new float[1 * 3 * h * w];
		int planeSize = h * w;
		for (int y = 0; y < h; y++) {
			int rowStart = y * w * 3;
			for (int x = 0; x < w; x++) {
				int px = rowStart + x * 3;
				float b = (raw[px] & 0xFF) - 127.5f;
				float g = (raw[px + 1] & 0xFF) - 127.5f;
				float r = (raw[px + 2] & 0xFF) - 127.5f;
				out[x + y * w] = b / 128.0f;                  // B plane
				out[planeSize + x + y * w] = g / 128.0f;      // G plane
				out[2 * planeSize + x + y * w] = r / 128.0f;  // R plane
			}
		}
		return out;
	}

	/**
	 * 把检测到的框从 letterbox 坐标系还原到原图坐标系。
	 */
	public static List<FaceBox> mapBoxesBackToOriginal(List<FaceBox> boxes, LetterboxResult lb) {
		List<FaceBox> out = new ArrayList<>(boxes.size());
		for (FaceBox b : boxes) {
			float x1 = (b.getX1() - lb.dx) / lb.scale;
			float y1 = (b.getY1() - lb.dy) / lb.scale;
			float x2 = (b.getX2() - lb.dx) / lb.scale;
			float y2 = (b.getY2() - lb.dy) / lb.scale;
			float[] lmk = null;
			float[] src = b.getLandmarks();
			if (src != null && src.length == 10) {
				lmk = new float[10];
				for (int i = 0; i < 10; i += 2) {
					lmk[i] = (src[i] - lb.dx) / lb.scale;
					lmk[i + 1] = (src[i + 1] - lb.dy) / lb.scale;
				}
			}
			out.add(new FaceBox(x1, y1, x2, y2, b.getScore(), lmk));
		}
		return out;
	}

	/**
	 * Letterbox 处理结果。
	 */
	public record LetterboxResult(BufferedImage image, float scale, int dx, int dy, int newW, int newH) {
	}
}
