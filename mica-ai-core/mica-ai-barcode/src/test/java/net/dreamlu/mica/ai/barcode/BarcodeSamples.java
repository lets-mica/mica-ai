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
package net.dreamlu.mica.ai.barcode;

import com.google.zxing.BarcodeFormat;
import com.google.zxing.EncodeHintType;
import com.google.zxing.MultiFormatWriter;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.qrcode.QRCodeWriter;
import com.google.zxing.qrcode.decoder.ErrorCorrectionLevel;
import net.dreamlu.mica.ai.barcode.model.BarcodeImage;

import java.util.EnumMap;
import java.util.Map;

/**
 * 与选型探针同款的全内存合成样本，用来把探针的关键结论固化成回归测试。
 */
final class BarcodeSamples {

	static final int WHITE = 255;
	static final String INVOICE = "INV-2026-1105-0001";

	private BarcodeSamples() {
	}

	static Gray qr(String text, int scale, int margin) throws Exception {
		return render(qrMatrix(text), scale, margin);
	}

	static Gray code128(String text) throws Exception {
		return render(oneDimensional(text, BarcodeFormat.CODE_128, 240, 64), 1, 0);
	}

	static Gray ean13(String text) throws Exception {
		return render(oneDimensional(text, BarcodeFormat.EAN_13, 200, 80), 1, 0);
	}

	static Gray compose(Gray[] parts, int cols, int gap) {
		int rows = (parts.length + cols - 1) / cols;
		int cellWidth = 0;
		int cellHeight = 0;
		for (Gray part : parts) {
			cellWidth = Math.max(cellWidth, part.w);
			cellHeight = Math.max(cellHeight, part.h);
		}
		Gray out = new Gray(cols * cellWidth + (cols + 1) * gap,
			rows * cellHeight + (rows + 1) * gap);
		for (int i = 0; i < parts.length; i++) {
			int ox = gap + (i % cols) * (cellWidth + gap);
			int oy = gap + (i / cols) * (cellHeight + gap);
			Gray part = parts[i];
			for (int y = 0; y < part.h; y++) {
				for (int x = 0; x < part.w; x++) {
					out.set(ox + x, oy + y, part.at(x, y));
				}
			}
		}
		return out;
	}

	static Gray blur(Gray src, double sigma) {
		int radius = (int) Math.ceil(3 * sigma);
		double[] kernel = new double[2 * radius + 1];
		double sum = 0d;
		for (int i = -radius; i <= radius; i++) {
			kernel[i + radius] = Math.exp(-i * i / (2 * sigma * sigma));
			sum += kernel[i + radius];
		}
		for (int i = 0; i < kernel.length; i++) {
			kernel[i] /= sum;
		}
		Gray horizontal = new Gray(src.w, src.h);
		for (int y = 0; y < src.h; y++) {
			for (int x = 0; x < src.w; x++) {
				double value = 0d;
				for (int i = -radius; i <= radius; i++) {
					int xx = Math.min(src.w - 1, Math.max(0, x + i));
					value += src.at(xx, y) * kernel[i + radius];
				}
				horizontal.set(x, y, clamp((int) Math.round(value)));
			}
		}
		Gray out = new Gray(src.w, src.h);
		for (int y = 0; y < src.h; y++) {
			for (int x = 0; x < src.w; x++) {
				double value = 0d;
				for (int i = -radius; i <= radius; i++) {
					int yy = Math.min(src.h - 1, Math.max(0, y + i));
					value += horizontal.at(x, yy) * kernel[i + radius];
				}
				out.set(x, y, clamp((int) Math.round(value)));
			}
		}
		return out;
	}

	static BarcodeImage image(Gray gray) {
		return BarcodeImage.ofGray(gray.w, gray.h, gray.p);
	}

	private static BitMatrix qrMatrix(String text) throws Exception {
		Map<EncodeHintType, Object> hints = new EnumMap<>(EncodeHintType.class);
		hints.put(EncodeHintType.CHARACTER_SET, "UTF-8");
		hints.put(EncodeHintType.ERROR_CORRECTION, ErrorCorrectionLevel.M);
		hints.put(EncodeHintType.MARGIN, 4);
		return new QRCodeWriter().encode(text, BarcodeFormat.QR_CODE, 0, 0, hints);
	}

	private static BitMatrix oneDimensional(String text, BarcodeFormat format,
											int width, int height) throws Exception {
		Map<EncodeHintType, Object> hints = new EnumMap<>(EncodeHintType.class);
		hints.put(EncodeHintType.CHARACTER_SET, "UTF-8");
		return new MultiFormatWriter().encode(text, format, width, height, hints);
	}

	private static Gray render(BitMatrix matrix, int scale, int margin) {
		int width = matrix.getWidth() * scale + 2 * margin;
		int height = matrix.getHeight() * scale + 2 * margin;
		Gray gray = new Gray(width, height);
		for (int y = 0; y < matrix.getHeight(); y++) {
			for (int x = 0; x < matrix.getWidth(); x++) {
				if (!matrix.get(x, y)) {
					continue;
				}
				for (int dy = 0; dy < scale; dy++) {
					for (int dx = 0; dx < scale; dx++) {
						gray.set(margin + x * scale + dx, margin + y * scale + dy, 0);
					}
				}
			}
		}
		return gray;
	}

	private static int clamp(int value) {
		return value < 0 ? 0 : (value > 255 ? 255 : value);
	}

	static final class Gray {

		final int w;
		final int h;
		final int[] p;

		Gray(int w, int h) {
			this.w = w;
			this.h = h;
			this.p = new int[w * h];
			java.util.Arrays.fill(p, WHITE);
		}

		int at(int x, int y) {
			return p[y * w + x];
		}

		void set(int x, int y, int value) {
			p[y * w + x] = value;
		}
	}
}
