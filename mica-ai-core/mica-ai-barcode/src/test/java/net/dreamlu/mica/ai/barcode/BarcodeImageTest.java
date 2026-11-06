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

import net.dreamlu.mica.ai.barcode.model.BarcodeImage;
import net.dreamlu.mica.ai.barcode.util.BarcodeImageUtils;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class BarcodeImageTest {

	@Test
	void grayAndBgrFactoriesShouldProduceArgbPixels() {
		BarcodeImage gray = BarcodeImage.ofGray(2, 1, new int[]{0, 300});
		assertThat(gray.getArgb()).containsExactly(0xFF000000, 0xFFFFFFFF);

		BarcodeImage grayBytes = BarcodeImage.ofGrayBytes(1, 1, new byte[]{(byte) 0x80});
		assertThat(grayBytes.getArgb()).containsExactly(0xFF808080);

		BarcodeImage bgr = BarcodeImage.ofBgr(1, 1, new byte[]{10, 20, 30});
		assertThat(bgr.getArgb()).containsExactly(0xFF1E140A);
		assertThat(bgr.pixel(0, 0)).isEqualTo(0xFF1E140A);
	}

	@Test
	void ofArgbShouldDefensivelyCopy() {
		int[] source = {0xFF000000, 0xFFFFFFFF};
		BarcodeImage image = BarcodeImage.ofArgb(2, 1, source);
		source[0] = 0xFF123456;

		assertThat(image.getArgb()[0]).isEqualTo(0xFF000000);
	}

	@Test
	void illegalPixelsShouldFailFast() {
		assertThatThrownBy(() -> BarcodeImage.ofGray(2, 2, new int[]{0}))
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("width*height");
		assertThatThrownBy(() -> BarcodeImage.ofBgr(2, 2, new byte[]{0}))
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("width*height*3");
		assertThatThrownBy(() -> BarcodeImage.ofGray(0, 1, new int[0]))
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("宽高");
	}

	@Test
	void nearestUpscaleShouldDuplicatePixelsWithoutInterpolation() {
		BarcodeImage src = BarcodeImage.ofGray(2, 2, new int[]{0, 255, 255, 0});

		BarcodeImage upscaled = BarcodeImageUtils.upscaleNearest(src, 2);

		assertThat(upscaled.getWidth()).isEqualTo(4);
		assertThat(upscaled.getHeight()).isEqualTo(4);
		assertThat(upscaled.getArgb()).containsExactly(
			0xFF000000, 0xFF000000, 0xFFFFFFFF, 0xFFFFFFFF,
			0xFF000000, 0xFF000000, 0xFFFFFFFF, 0xFFFFFFFF,
			0xFFFFFFFF, 0xFFFFFFFF, 0xFF000000, 0xFF000000,
			0xFFFFFFFF, 0xFFFFFFFF, 0xFF000000, 0xFF000000);
	}

	@Test
	void cropShouldKeepCorrectPixelsAndBounds() {
		BarcodeImage src = BarcodeImage.ofGray(3, 3,
			new int[]{1, 2, 3, 4, 5, 6, 7, 8, 9});

		BarcodeImage cropped = BarcodeImageUtils.crop(src, 1, 1, 5, 5);

		assertThat(cropped.getWidth()).isEqualTo(2);
		assertThat(cropped.getHeight()).isEqualTo(2);
		assertThat(cropped.getArgb()).containsExactly(0xFF050505, 0xFF060606, 0xFF080808, 0xFF090909);
	}

	@Test
	void encodedBytesShouldRoundTripThroughImageIo() throws Exception {
		BufferedImage buffered = new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB);
		for (int y = 0; y < 2; y++) {
			for (int x = 0; x < 3; x++) {
				buffered.setRGB(x, y, (x == 0 && y == 0) ? 0x000000 : 0xFFFFFF);
			}
		}
		ByteArrayOutputStream out = new ByteArrayOutputStream();
		assertThat(ImageIO.write(buffered, "png", out)).isTrue();

		BarcodeImage image = BarcodeImage.fromEncodedBytes(out.toByteArray());

		assertThat(image.getWidth()).isEqualTo(3);
		assertThat(image.getHeight()).isEqualTo(2);
		assertThat(image.getArgb()[0]).isEqualTo(0xFF000000);
		assertThat(image.getArgb()[1]).isEqualTo(0xFFFFFFFF);
	}

	@Test
	void undecodableBytesShouldFailFast() {
		assertThatThrownBy(() -> BarcodeImage.fromEncodedBytes(new byte[]{1, 2, 3}))
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("无法识别");
		assertThatThrownBy(() -> BarcodeImage.fromEncodedBytes(new byte[0]))
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("imageBytes");
	}
}
