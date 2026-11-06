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
package net.dreamlu.mica.ai.barcode.model;

import lombok.Getter;
import lombok.ToString;

import java.util.Arrays;

/**
 * 解码输入：一张 JVM 内存里的 ARGB 像素图（无 {@code Mat}、无 {@code BufferedImage} 类型泄漏）。
 *
 * <p>像素数组是<b>只读约定</b>：构造时会复制入参，但 {@link #getArgb()} 返回内部数组以避免
 * 解码链反复拷贝大图；调用方不得修改其内容。所有工厂方法都做长度与尺寸校验，非法输入立即抛
 * {@link IllegalArgumentException}。
 *
 * <p>{@link #fromEncodedBytes(byte[])} 是便捷入口（内部走 JDK 自带 ImageIO，headless 安全），
 * 只被真正调用时才会加载 {@code java.desktop}；精简 JRE 请用像素工厂。
 */
@Getter
@ToString(exclude = "argb")
public final class BarcodeImage {

	private final int width;
	private final int height;
	private final int[] argb;

	private BarcodeImage(int width, int height, int[] argb) {
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("图像宽高必须为正数: " + width + "x" + height);
		}
		long expected = (long) width * height;
		if (argb == null || argb.length != expected) {
			throw new IllegalArgumentException("像素数组长度必须等于 width*height: " + expected);
		}
		this.width = width;
		this.height = height;
		this.argb = argb;
	}

	public static BarcodeImage ofArgb(int width, int height, int[] argb) {
		if (argb == null) {
			throw new IllegalArgumentException("argb 不能为 null");
		}
		return new BarcodeImage(width, height, Arrays.copyOf(argb, argb.length));
	}

	public static BarcodeImage ofGray(int width, int height, int[] gray) {
		return fromGray(width, height, gray, false);
	}

	public static BarcodeImage ofGrayBytes(int width, int height, byte[] gray) {
		if (gray == null) {
			throw new IllegalArgumentException("gray 不能为 null");
		}
		int[] pixels = new int[gray.length];
		for (int i = 0; i < gray.length; i++) {
			pixels[i] = gray[i] & 0xFF;
		}
		return fromGray(width, height, pixels, true);
	}

	/**
	 * 从 OpenCV {@code Mat.get(0, 0, byte[])} 取出的 BGR 交错字节构造，调用方负责保证通道数为 3。
	 */
	public static BarcodeImage ofBgr(int width, int height, byte[] bgr) {
		if (bgr == null) {
			throw new IllegalArgumentException("bgr 不能为 null");
		}
		long expected = (long) width * height * 3;
		if (bgr.length != expected) {
			throw new IllegalArgumentException("BGR 字节长度必须等于 width*height*3: " + expected);
		}
		int[] pixels = new int[width * height];
		for (int i = 0; i < pixels.length; i++) {
			int b = bgr[i * 3] & 0xFF;
			int g = bgr[i * 3 + 1] & 0xFF;
			int r = bgr[i * 3 + 2] & 0xFF;
			pixels[i] = 0xFF000000 | (r << 16) | (g << 8) | b;
		}
		return new BarcodeImage(width, height, pixels);
	}

	public static BarcodeImage fromEncodedBytes(byte[] imageBytes) {
		return BarcodeImageIO.read(imageBytes);
	}

	public int pixel(int x, int y) {
		return argb[y * width + x];
	}

	private static BarcodeImage fromGray(int width, int height, int[] gray, boolean trusted) {
		if (gray == null) {
			throw new IllegalArgumentException("gray 不能为 null");
		}
		int[] pixels = new int[gray.length];
		for (int i = 0; i < gray.length; i++) {
			int value = trusted ? gray[i] : clamp(gray[i]);
			pixels[i] = 0xFF000000 | (value << 16) | (value << 8) | value;
		}
		return new BarcodeImage(width, height, pixels);
	}

	private static int clamp(int value) {
		return value < 0 ? 0 : (Math.min(value, 255));
	}
}
