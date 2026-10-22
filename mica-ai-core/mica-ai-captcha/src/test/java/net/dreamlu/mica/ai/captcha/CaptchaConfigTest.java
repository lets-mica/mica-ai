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

import net.dreamlu.mica.ai.captcha.config.CaptchaConfig;
import net.dreamlu.mica.ai.captcha.config.CaptchaInterpolation;
import net.dreamlu.mica.ai.captcha.config.CaptchaNormalization;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link CaptchaConfig} 校验逻辑单元测试（不依赖模型文件）。
 */
class CaptchaConfigTest {

	@Test
	void blankModelPathShouldFailFast() {
		CaptchaConfig config = CaptchaConfig.builder()
			.charsetPath("charset.txt")
			.build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}

	@Test
	void blankCharsetPathShouldFailFast() {
		CaptchaConfig config = CaptchaConfig.builder()
			.modelPath("model.onnx")
			.build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("charsetPath");
	}

	@Test
	void nonPositiveInputHeightShouldFailFast() {
		CaptchaConfig config = CaptchaConfig.builder()
			.modelPath("model.onnx")
			.charsetPath("charset.txt")
			.inputHeight(0)
			.build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputHeight");
	}

	@Test
	void missingModelFileShouldFailFast() {
		CaptchaConfig config = CaptchaConfig.builder()
			.modelPath("no-such-dir/no-such-model.onnx")
			.charsetPath("no-such-dir/no-such-charset.txt")
			.build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("不存在");
	}

	@Test
	void defaultsShouldMatchOfficialModelContract() {
		CaptchaConfig config = CaptchaConfig.builder()
			.modelPath("model.onnx")
			.charsetPath("charset.txt")
			.build();
		assertThat(config.getInputHeight()).isEqualTo(64);
		assertThat(config.getNormalization()).isEqualTo(CaptchaNormalization.DIV_255);
		assertThat(config.getInterpolation()).isEqualTo(CaptchaInterpolation.BILINEAR);
		assertThat(config.isPngFix()).isTrue();
		assertThat(config.getOnnx()).isNotNull();
	}

	@Test
	void toBuilderShouldKeepOtherFields() {
		CaptchaConfig config = CaptchaConfig.builder()
			.modelPath("model.onnx")
			.charsetPath("charset.txt")
			.inputHeight(32)
			.build();
		CaptchaConfig derived = config.toBuilder().inputHeight(48).build();
		assertThat(derived.getInputHeight()).isEqualTo(48);
		assertThat(derived.getModelPath()).isEqualTo("model.onnx");
		assertThat(derived.getCharsetPath()).isEqualTo("charset.txt");
	}
}
