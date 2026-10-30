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
package net.dreamlu.mica.ai.enhance.config;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class EnhanceConfigTest {

	@Test
	void defaultsAreProductionReady() {
		EnhanceConfig config = EnhanceConfig.builder().build();
		assertThat(config.getModelVersion()).isEqualTo("realesr-general-x4v3");
		assertThat(config.getScale()).isEqualTo(4);
		assertThat(config.getTileSize()).isEqualTo(256);
		assertThat(config.getTileOverlap()).isEqualTo(16);
		assertThat(config.getOnnx()).isNotNull();
		config.validate();
	}

	@Test
	void resolvesBundledClasspathModel() {
		EnhanceConfig config = EnhanceConfig.builder().build();
		assertThat(config.resolveModelPath())
			.isEqualTo("classpath:mica-ai/models/enhance/realesr-general-x4v3.onnx");
	}

	@Test
	void resolvesExplicitModelPath() {
		EnhanceConfig config = EnhanceConfig.builder()
			.modelPath("/data/models/realesr-general-x4v3.onnx")
			.modelVersion("ignored")
			.build();
		assertThat(config.resolveModelPath()).isEqualTo("/data/models/realesr-general-x4v3.onnx");
	}

	@Test
	void rejectsEmptyModelVersion() {
		EnhanceConfig config = EnhanceConfig.builder().modelVersion("").build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelVersion");
	}

	@Test
	void rejectsNonPositiveScale() {
		EnhanceConfig config = EnhanceConfig.builder().scale(0).build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("scale");
	}

	@Test
	void rejectsNonPositiveTileSize() {
		EnhanceConfig config = EnhanceConfig.builder().tileSize(0).build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("tileSize");
	}

	@Test
	void rejectsNegativeOverlap() {
		EnhanceConfig config = EnhanceConfig.builder().tileOverlap(-1).build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("tileOverlap");
	}

	@Test
	void rejectsOverlapNotSmallerThanTile() {
		EnhanceConfig config = EnhanceConfig.builder()
			.tileSize(32)
			.tileOverlap(32)
			.build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("tileOverlap");
	}
}
