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

import lombok.experimental.UtilityClass;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * 便捷字节入口的隔离层：{@code java.awt} / {@code javax.imageio} 类型只出现在本类的常量池与签名里，
 * 且本类为包内可见（禁止 public）。{@link BarcodeImage} 与 {@code BarcodeEngine} 的公开方法签名
 * 不含任何 AWT 类型，精简 JRE 只要不调用 {@code fromEncodedBytes} 就不会触发类加载。
 *
 * <p>{@code ImageIO.read} 只做内存解码、不接触显示设备，{@code java.awt.headless=true} 下可用；
 * 这里刻意不引 ZXing 的 {@code javase} 模块（其 {@code BufferedImageLuminanceSource} 会把 AWT
 * 类型扩散到整个解码链）。
 */
@UtilityClass
class BarcodeImageIO {

	static BarcodeImage read(byte[] imageBytes) {
		if (imageBytes == null || imageBytes.length == 0) {
			throw new IllegalArgumentException("imageBytes 不能为空");
		}
		BufferedImage image;
		try {
			image = ImageIO.read(new ByteArrayInputStream(imageBytes));
		} catch (IOException e) {
			throw new IllegalArgumentException("无法解码图像字节", e);
		}
		if (image == null) {
			throw new IllegalArgumentException("无法识别的图像格式（仅支持 ImageIO 可解码的 png / jpg / bmp 等）");
		}
		int width = image.getWidth();
		int height = image.getHeight();
		int[] argb = image.getRGB(0, 0, width, height, null, 0, width);
		return BarcodeImage.ofArgb(width, height, argb);
	}
}
