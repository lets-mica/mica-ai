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
package net.dreamlu.mica.ai.captcha.util;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.captcha.config.CaptchaInterpolation;
import net.dreamlu.mica.ai.captcha.config.CaptchaNormalization;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

/**
 * 验证码图片解码与预处理工具。
 *
 * <p>预处理链路与 ddddocr Python 原版严格对齐：
 * <ol>
 *   <li>可选的透明通道处理（PNG 透明像素按黑色合成）</li>
 *   <li>等比缩放到固定高度（宽度按原图比例自适应，整数截断）</li>
 *   <li>灰度化（ITU-R 601-2：{@code L = 0.299R + 0.587G + 0.114B}，同 PIL {@code convert('L')}）</li>
 *   <li>归一化</li>
 * </ol>
 *
 * <p>仅依赖 JDK 的 {@code javax.imageio} / {@code java.awt}，不引入 OpenCV 原生库。
 */
@UtilityClass
public class CaptchaImageUtils {

	/**
	 * 从字节数组解码图片。
	 *
	 * @param imageBytes 图片字节（png / jpg 等）
	 * @return 解码后的图片
	 * @throws MicaAiException {@link ErrorCode#MODEL_LOAD_FAILED} 解码失败
	 */
	public static BufferedImage loadImage(byte[] imageBytes) {
		if (imageBytes == null || imageBytes.length == 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "图片字节为空");
		}
		try (ByteArrayInputStream in = new ByteArrayInputStream(imageBytes)) {
			BufferedImage image = ImageIO.read(in);
			if (image == null) {
				throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
					"无法解码图片：不是受支持的图片格式");
			}
			return image;
		} catch (IOException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "读取图片失败", e);
		}
	}

	/**
	 * 从文件路径解码图片。
	 *
	 * @param imagePath 图片文件路径
	 * @return 解码后的图片
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#MODEL_LOAD_FAILED} 解码失败
	 */
	public static BufferedImage loadImage(String imagePath) {
		Path path = Paths.get(imagePath);
		if (!Files.exists(path)) {
			throw new MicaAiException(ErrorCode.NOT_FOUND, "图片文件不存在: " + imagePath);
		}
		try {
			return loadImage(Files.readAllBytes(path));
		} catch (IOException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"读取图片文件失败: " + imagePath, e);
		}
	}

	/**
	 * 从 base64 字符串解码图片（自动剥离 {@code data:image/png;base64,} 前缀）。
	 *
	 * @param base64Image base64 图片字符串
	 * @return 解码后的图片
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 参数为空；
	 *                         {@link ErrorCode#MODEL_LOAD_FAILED} 解码失败
	 */
	public static BufferedImage loadBase64Image(String base64Image) {
		if (base64Image == null || base64Image.trim().isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "base64 图片不能为空");
		}
		String data = base64Image;
		int comma = base64Image.indexOf(',');
		if (comma > 0) {
			data = base64Image.substring(comma + 1);
		}
		try {
			return loadImage(Base64.getDecoder().decode(data.trim()));
		} catch (IllegalArgumentException e) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "base64 图片不是合法的 Base64", e);
		}
	}

	/**
	 * 预处理为 ONNX 输入张量（{@code [1, 1, H, W]}）。
	 *
	 * @param image         输入图片
	 * @param targetHeight  目标高度，宽度按原图比例自适应
	 * @param interpolation 缩放插值方式
	 * @param normalization 归一化方式
	 * @param pngFix        是否把透明像素按黑色合成
	 * @return 预处理后的张量数据
	 */
	public static ImageTensor prepareForOnnx(BufferedImage image, int targetHeight,
											 CaptchaInterpolation interpolation,
											 CaptchaNormalization normalization,
											 boolean pngFix) {
		BufferedImage source = pngFix ? flattenAlpha(image) : image;
		BufferedImage resized = resizeKeepAspect(source, targetHeight, interpolation);
		int width = resized.getWidth();
		int height = resized.getHeight();
		float[] data = new float[height * width];
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int rgb = resized.getRGB(x, y);
				int r = (rgb >> 16) & 0xFF;
				int g = (rgb >> 8) & 0xFF;
				int b = rgb & 0xFF;
				int gray = (int) (0.299 * r + 0.587 * g + 0.114 * b);
				data[y * width + x] = normalization.apply(gray / 255.0f);
			}
		}
		return new ImageTensor(data, width, height);
	}

	/**
	 * 等比缩放到目标高度（宽度按比例取整，最小 1 像素）。
	 *
	 * @param image         原图
	 * @param targetHeight  目标高度
	 * @param interpolation 插值方式
	 * @return 缩放后的图片（无透明通道）
	 */
	public static BufferedImage resizeKeepAspect(BufferedImage image, int targetHeight,
												 CaptchaInterpolation interpolation) {
		int originalWidth = image.getWidth();
		int originalHeight = image.getHeight();
		int targetWidth = (int) (originalWidth * ((double) targetHeight / originalHeight));
		if (targetWidth < 1) {
			targetWidth = 1;
		}
		BufferedImage resized = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = resized.createGraphics();
		try {
			g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, interpolation.getAwtValue());
			g.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g.drawImage(image, 0, 0, targetWidth, targetHeight, null);
		} finally {
			g.dispose();
		}
		return resized;
	}

	/**
	 * 把带透明通道的图片按黑色背景合成，返回无 alpha 的 RGB 图片。
	 *
	 * <p>alpha=0 的像素变为纯黑；0&lt;alpha&lt;255 的像素与黑色按 alpha 线性混合；
	 * alpha=255 的像素保持原样。
	 *
	 * @param image 原图
	 * @return 无透明通道的图片
	 */
	public static BufferedImage flattenAlpha(BufferedImage image) {
		if (!image.getColorModel().hasAlpha()) {
			return image;
		}
		int width = image.getWidth();
		int height = image.getHeight();
		BufferedImage rgb = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int argb = image.getRGB(x, y);
				int alpha = (argb >>> 24) & 0xFF;
				if (alpha == 0xFF) {
					rgb.setRGB(x, y, argb & 0x00FFFFFF);
				} else if (alpha == 0) {
					rgb.setRGB(x, y, 0x000000);
				} else {
					int r = (argb >> 16) & 0xFF;
					int g = (argb >> 8) & 0xFF;
					int b = argb & 0xFF;
					int nr = r * alpha / 255;
					int ng = g * alpha / 255;
					int nb = b * alpha / 255;
					rgb.setRGB(x, y, (nr << 16) | (ng << 8) | nb);
				}
			}
		}
		return rgb;
	}

	/**
	 * 预处理后的图像张量（CHW 顺序的一维数组 + 尺寸）。
	 */
	@Getter
	@RequiredArgsConstructor
	public static final class ImageTensor {
		/**
		 * 张量数据，布局为
		 */
		private final float[] data;
		/**
		 * 张量宽度
		 */
		private final int width;
		/**
		 * 张量高度
		 */
		private final int height;
	}
}
