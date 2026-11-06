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

import net.dreamlu.mica.ai.barcode.config.BarcodeConfig;
import net.dreamlu.mica.ai.barcode.core.BarcodeEngine;
import net.dreamlu.mica.ai.barcode.model.BarcodeImage;
import net.dreamlu.mica.ai.barcode.model.BarcodeResult;
import net.dreamlu.mica.ai.barcode.model.BarcodeSymbology;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 把选型实测结论固化为回归测试：
 * 多尺度救 1px 小码与模糊、分块救一图多码、一维码必须原分辨率、
 * 中文 QR 以 zxing 为准（不走 OpenCV ECI 兜底）。
 */
class BarcodeEngineTest {

	@Test
	void cleanQrShouldDecodeDirectly() throws Exception {
		BarcodeImage image = BarcodeSamples.image(BarcodeSamples.qr(BarcodeSamples.INVOICE, 4, 8));

		Optional<BarcodeResult> result = BarcodeEngine.defaults().decodeFirst(image);

		assertThat(result).isPresent();
		assertThat(result.get().getText()).isEqualTo(BarcodeSamples.INVOICE);
		assertThat(result.get().getSymbology()).isEqualTo(BarcodeSymbology.QR_CODE);
		assertThat(result.get().getStrategy()).isEqualTo("DIRECT");
		assertThat(result.get().hasBoundingBox()).isTrue();
		int[] box = result.get().getBoundingBox();
		assertThat(box[0]).isGreaterThanOrEqualTo(0).isLessThanOrEqualTo(image.getWidth());
		assertThat(box[1]).isGreaterThanOrEqualTo(0).isLessThanOrEqualTo(image.getHeight());
		assertThat(box[2]).isGreaterThanOrEqualTo(box[0]);
		assertThat(box[3]).isGreaterThanOrEqualTo(box[1]);
	}

	@Test
	void onePixelQrNeedsMultiScaleRetry() throws Exception {
		BarcodeImage tiny = BarcodeSamples.image(BarcodeSamples.qr(BarcodeSamples.INVOICE, 1, 4));
		BarcodeConfig directOnly = BarcodeConfig.builder()
			.multiScale(false)
			.tileScan(false)
			.globalHistogramFallback(false)
			.build();

		assertThat(BarcodeEngine.create(directOnly).decodeFirst(tiny)).isEmpty();

		Optional<BarcodeResult> result = BarcodeEngine.defaults().decodeFirst(tiny);
		assertThat(result).isPresent();
		assertThat(result.get().getText()).isEqualTo(BarcodeSamples.INVOICE);
		assertThat(result.get().getStrategy()).startsWith("UPSCALE_x");
	}

	@Test
	void blurredQrShouldDecodeWithFullChain() throws Exception {
		BarcodeImage blurred = BarcodeSamples.image(
			BarcodeSamples.blur(BarcodeSamples.qr(BarcodeSamples.INVOICE, 4, 8), 2.0));

		Optional<BarcodeResult> result = BarcodeEngine.defaults().decodeFirst(blurred);

		assertThat(result).isPresent();
		assertThat(result.get().getText()).isEqualTo(BarcodeSamples.INVOICE);
	}

	@Test
	void fourCodesNeedsTileScan() throws Exception {
		String[] texts = {"MICA-A", "MICA-B", "MICA-C", "MICA-D"};
		BarcodeSamples.Gray[] parts = new BarcodeSamples.Gray[texts.length];
		for (int i = 0; i < texts.length; i++) {
			parts[i] = BarcodeSamples.qr(texts[i], 3, 6);
		}
		BarcodeImage grid = BarcodeSamples.image(BarcodeSamples.compose(parts, 2, 12));

		List<BarcodeResult> all = BarcodeEngine.defaults().decode(grid);
		assertThat(all).extracting(BarcodeResult::getText)
			.containsExactlyInAnyOrder(texts);
		assertThat(all).extracting(BarcodeResult::getStrategy)
			.anyMatch(strategy -> strategy.startsWith("TILE_"));

		BarcodeConfig noTile = BarcodeConfig.builder().tileScan(false).build();
		assertThat(BarcodeEngine.create(noTile).decode(grid).size()).isLessThan(texts.length);
	}

	@Test
	void oneDimensionalCodesShouldDecodeAtNativeResolution() throws Exception {
		BarcodeEngine engine = BarcodeEngine.defaults();

		Optional<BarcodeResult> code128 = engine.decodeFirst(
			BarcodeSamples.image(BarcodeSamples.code128("MICA-1001")));
		assertThat(code128).isPresent();
		assertThat(code128.get().getText()).isEqualTo("MICA-1001");
		assertThat(code128.get().getSymbology()).isEqualTo(BarcodeSymbology.CODE_128);

		Optional<BarcodeResult> ean13 = engine.decodeFirst(
			BarcodeSamples.image(BarcodeSamples.ean13("5901234123457")));
		assertThat(ean13).isPresent();
		assertThat(ean13.get().getText()).isEqualTo("5901234123457");
		assertThat(ean13.get().getSymbology()).isEqualTo(BarcodeSymbology.EAN_13);
	}

	@Test
	void chineseQrShouldKeepUtf8Text() throws Exception {
		String text = "发票号码:144031609110";

		Optional<BarcodeResult> result = BarcodeEngine.defaults().decodeFirst(
			BarcodeSamples.image(BarcodeSamples.qr(text, 4, 8)));

		assertThat(result).isPresent();
		assertThat(result.get().getText()).isEqualTo(text);
		assertThat(result.get().getCharset()).isEqualTo("UTF-8");
	}

	@Test
	void possibleFormatsShouldRestrictSymbologies() throws Exception {
		BarcodeConfig qrOnly = BarcodeConfig.builder()
			.formats(Collections.singletonList(BarcodeSymbology.QR_CODE))
			.build();

		Optional<BarcodeResult> result = BarcodeEngine.create(qrOnly).decodeFirst(
			BarcodeSamples.image(BarcodeSamples.code128("MICA-1001")));

		assertThat(result).isEmpty();
	}

	@Test
	void decodeBytesShouldAcceptPng() throws Exception {
		BarcodeSamples.Gray gray = BarcodeSamples.qr("MICA-BYTES", 4, 8);
		BufferedImage buffered = new BufferedImage(gray.w, gray.h, BufferedImage.TYPE_INT_RGB);
		for (int y = 0; y < gray.h; y++) {
			for (int x = 0; x < gray.w; x++) {
				int value = gray.at(x, y);
				buffered.setRGB(x, y, (value << 16) | (value << 8) | value);
			}
		}
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		assertThat(ImageIO.write(buffered, "png", out)).isTrue();

		Optional<BarcodeResult> result = BarcodeEngine.defaults().decodeFirstBytes(out.toByteArray());

		assertThat(result).isPresent();
		assertThat(result.get().getText()).isEqualTo("MICA-BYTES");
	}

	@Test
	void nullImageShouldReturnEmptyResult() {
		BarcodeEngine engine = BarcodeEngine.defaults();

		assertThat(engine.decode(null)).isEmpty();
		assertThat(engine.decodeFirst(null)).isEmpty();
		assertThat(engine.formats()).isEqualTo(BarcodeSymbology.DEFAULT_FORMATS);
	}
}
