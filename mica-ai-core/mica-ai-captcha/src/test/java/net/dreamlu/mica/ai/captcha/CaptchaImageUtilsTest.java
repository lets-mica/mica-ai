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
package net.dreamlu.mica.ai.captcha;

import net.dreamlu.mica.ai.captcha.charset.CharsetManager;
import net.dreamlu.mica.ai.captcha.config.CaptchaInterpolation;
import net.dreamlu.mica.ai.captcha.config.CaptchaNormalization;
import net.dreamlu.mica.ai.captcha.util.CaptchaImageUtils;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.Test;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 图像预处理与字符集加载的单元测试（不依赖 ONNX 模型）。
 */
class CaptchaImageUtilsTest {

	@Test
	void prepareForOnnxShouldKeepAspectRatioAndHeight() {
		BufferedImage image = new BufferedImage(100, 36, BufferedImage.TYPE_INT_RGB);
		CaptchaImageUtils.ImageTensor tensor = CaptchaImageUtils.prepareForOnnx(
			image, 64, CaptchaInterpolation.BILINEAR, CaptchaNormalization.DIV_255, true);

		assertThat(tensor.getHeight()).isEqualTo(64);
		// 宽度按 100 * 64 / 36 = 177.7 截断
		assertThat(tensor.getWidth()).isEqualTo(177);
		assertThat(tensor.getData()).hasSize(64 * 177);
	}

	@Test
	void div255ShouldMapToUnitRange() {
		BufferedImage image = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
		CaptchaImageUtils.ImageTensor tensor = CaptchaImageUtils.prepareForOnnx(
			image, 4, CaptchaInterpolation.NEAREST, CaptchaNormalization.DIV_255, false);
		for (float v : tensor.getData()) {
			assertThat(v).isBetween(0f, 1f);
		}
	}

	@Test
	void mean05ShouldMapToSignedRange() {
		BufferedImage black = new BufferedImage(4, 4, BufferedImage.TYPE_INT_RGB);
		CaptchaImageUtils.ImageTensor tensor = CaptchaImageUtils.prepareForOnnx(
			black, 4, CaptchaInterpolation.NEAREST, CaptchaNormalization.MEAN_0_5, false);
		for (float v : tensor.getData()) {
			assertThat(v).isBetween(-1f, 1f);
			assertThat(v).isEqualTo(-1f);
		}
	}

	@Test
	void grayscaleShouldUseLumaFormula() {
		BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = image.createGraphics();
		g.setColor(new Color(255, 0, 0));
		g.fillRect(0, 0, 2, 2);
		g.dispose();

		CaptchaImageUtils.ImageTensor tensor = CaptchaImageUtils.prepareForOnnx(
			image, 2, CaptchaInterpolation.NEAREST, CaptchaNormalization.DIV_255, false);

		// L = 0.299 * 255 = 76.2 -> 76 ; 76 / 255 = 0.298
		assertThat(tensor.getData()[0]).isCloseTo(76f / 255f, org.assertj.core.data.Offset.offset(0.002f));
	}

	@Test
	void flattenAlphaShouldTurnTransparentPixelsBlack() {
		BufferedImage argb = new BufferedImage(2, 2, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g = argb.createGraphics();
		g.setColor(new Color(255, 0, 0, 0));
		g.fillRect(0, 0, 2, 2);
		g.dispose();

		BufferedImage flat = CaptchaImageUtils.flattenAlpha(argb);

		assertThat(flat.getType()).isEqualTo(BufferedImage.TYPE_INT_RGB);
		assertThat(flat.getRGB(0, 0) & 0x00FFFFFF).isEqualTo(0x000000);
	}

	@Test
	void loadBase64ShouldStripDataUriPrefix() {
		BufferedImage image = new BufferedImage(8, 8, BufferedImage.TYPE_INT_RGB);
		java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
		try {
			 javax.imageio.ImageIO.write(image, "png", out);
		} catch (java.io.IOException e) {
			throw new IllegalStateException(e);
		}
		String base64 = "data:image/png;base64,"
			+ java.util.Base64.getEncoder().encodeToString(out.toByteArray());

		BufferedImage decoded = CaptchaImageUtils.loadBase64Image(base64);

		assertThat(decoded).isNotNull();
		assertThat(decoded.getWidth()).isEqualTo(8);
	}

	@Test
	void loadBase64WithInvalidPayloadShouldFailFast() {
		assertThatThrownBy(() -> CaptchaImageUtils.loadBase64Image("not-base64!!!"))
			.isInstanceOf(MicaAiException.class);
	}

	@Test
	void charsetShouldKeepOneIndexPerCharacter() throws Exception {
		Path file = Files.createTempFile("captcha-charset", ".txt");
		Files.write(file, "ABC\n".getBytes(StandardCharsets.UTF_8));
		try {
			CharsetManager manager = CharsetManager.load(file.toString());
			// blank + A + B + C（换行不占索引）
			assertThat(manager.size()).isEqualTo(4);
			assertThat(manager.charAt(0)).isEmpty();
			assertThat(manager.charAt(1)).isEqualTo("A");
			assertThat(manager.charAt(3)).isEqualTo("C");
			assertThat(manager.isValidIndex(4)).isFalse();
		} finally {
			Files.deleteIfExists(file);
		}
	}

	@Test
	void charsetShouldNotDeduplicateRepeatedCharacters() throws Exception {
		Path file = Files.createTempFile("captcha-charset-dup", ".txt");
		Files.write(file, "ABA".getBytes(StandardCharsets.UTF_8));
		try {
			CharsetManager manager = CharsetManager.load(file.toString());
			// 去重会让后续索引整体错位，因此必须原样保留
			assertThat(manager.size()).isEqualTo(4);
			assertThat(manager.charAt(1)).isEqualTo("A");
			assertThat(manager.charAt(2)).isEqualTo("B");
			assertThat(manager.charAt(3)).isEqualTo("A");
		} finally {
			Files.deleteIfExists(file);
		}
	}

	@Test
	void charsetMissingFileShouldFailFast() {
		assertThatThrownBy(() -> CharsetManager.load(Paths.get("no-such-charset.txt").toString()))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("不存在");
	}
}
