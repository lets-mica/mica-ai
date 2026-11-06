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
package net.dreamlu.mica.ai.barcode.util;

import com.google.zxing.LuminanceSource;
import com.google.zxing.RGBLuminanceSource;
import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.barcode.model.BarcodeImage;

/**
 * 全内存像素变换。放大用<b>最近邻</b>：条码是二值图形，双线性会糊掉模块边缘，
 * 实测最近邻 ×2/×3 能把 1px 小码从 0% 拉到 100%。
 */
@UtilityClass
public class BarcodeImageUtils {
	private static final int WHITE = 0xFFFFFFFF;

	public static LuminanceSource toLuminanceSource(BarcodeImage image) {
		return new RGBLuminanceSource(image.getWidth(), image.getHeight(), image.getArgb());
	}

	public static BarcodeImage upscaleNearest(BarcodeImage src, int factor) {
		int sw = src.getWidth();
		int sh = src.getHeight();
		int dw = sw * factor;
		int dh = sh * factor;
		int[] srcPixels = src.getArgb();
		int[] dst = new int[dw * dh];
		for (int y = 0; y < dh; y++) {
			int sy = (int) Math.round((y + 0.5) * sh / (double) dh - 0.5);
			for (int x = 0; x < dw; x++) {
				int sx = (int) Math.round((x + 0.5) * sw / (double) dw - 0.5);
				dst[y * dw + x] = (sx >= 0 && sy >= 0 && sx < sw && sy < sh)
					? srcPixels[sy * sw + sx] : WHITE;
			}
		}
		return BarcodeImage.ofArgb(dw, dh, dst);
	}

	public static BarcodeImage crop(BarcodeImage src, int x1, int y1, int x2, int y2) {
		int left = Math.max(0, Math.min(x1, src.getWidth()));
		int top = Math.max(0, Math.min(y1, src.getHeight()));
		int right = Math.max(left, Math.min(x2, src.getWidth()));
		int bottom = Math.max(top, Math.min(y2, src.getHeight()));
		int width = right - left;
		int height = bottom - top;
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("裁剪区域为空: [" + x1 + "," + y1 + "," + x2 + "," + y2 + "]");
		}
		int[] dst = new int[width * height];
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				dst[y * width + x] = src.pixel(left + x, top + y);
			}
		}
		return BarcodeImage.ofArgb(width, height, dst);
	}
}
