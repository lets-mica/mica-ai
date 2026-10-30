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
package net.dreamlu.mica.ai.example.enhance;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.example.pipeline.enhance.EnhanceUpscaleOptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 图像超分参数校验单测（不需要模型）。
 */
class EnhanceUpscaleOptionsTest {

	@Test
	@DisplayName("默认值：4 倍输出、256 分块、1200 输入长边上限")
	void defaultsShouldMatchDesign() {
		EnhanceUpscaleOptions options = EnhanceUpscaleOptions.builder().build();
		assertThat(options.getOutScale()).isEqualTo(4);
		assertThat(options.getTileSize()).isEqualTo(256);
		assertThat(options.getMaxInputSide()).isEqualTo(1200);
	}

	@Test
	@DisplayName("输出倍率不能超过模型原生倍率")
	void outScaleShouldNotExceedModelScale() {
		EnhanceUpscaleOptions options = EnhanceUpscaleOptions.builder().outScale(5).build();
		assertThatThrownBy(() -> options.validate(4))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("outScale");
	}

	@Test
	@DisplayName("输出倍率不能小于 1")
	void outScaleShouldBePositive() {
		EnhanceUpscaleOptions options = EnhanceUpscaleOptions.builder().outScale(0).build();
		assertThatThrownBy(() -> options.validate(4))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("outScale");
	}

	@Test
	@DisplayName("分块边长越界应被拒绝")
	void tileSizeShouldBeWithinRange() {
		assertThatThrownBy(() -> EnhanceUpscaleOptions.builder().tileSize(16).build().validate(4))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("tileSize");
		assertThatThrownBy(() -> EnhanceUpscaleOptions.builder().tileSize(4096).build().validate(4))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("tileSize");
	}

	@Test
	@DisplayName("输入长边上限过小应被拒绝")
	void maxInputSideShouldNotBeTooSmall() {
		assertThatThrownBy(() -> EnhanceUpscaleOptions.builder().maxInputSide(32).build().validate(4))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("maxInputSide");
	}

	@Test
	@DisplayName("合法参数应通过校验")
	void validOptionsShouldPass() {
		EnhanceUpscaleOptions.builder().outScale(2).tileSize(128).maxInputSide(800).build().validate(4);
	}

}
