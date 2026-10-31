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

	private static final String MODEL = "model-tools/enhance/models/realesr-general-x4v3.onnx";

	@Test
	void defaultsAreProductionReady() {
		EnhanceConfig config = EnhanceConfig.builder().modelPath(MODEL).build();
		assertThat(config.getModelVersion()).isEqualTo("realesr-general-x4v3");
		assertThat(config.getScale()).isEqualTo(4);
		assertThat(config.getTileSize()).isEqualTo(256);
		assertThat(config.getTileOverlap()).isEqualTo(16);
		assertThat(config.getOnnx()).isNotNull();
		config.validate();
	}

	@Test
	void resolvesExplicitClasspathModel() {
		EnhanceConfig config = EnhanceConfig.builder()
			.modelPath("classpath:models/enhance/realesr-general-x4v3.onnx")
			.build();
		assertThat(config.resolveModelPath())
			.isEqualTo("classpath:models/enhance/realesr-general-x4v3.onnx");
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
	void rejectsMissingModelPath() {
		assertThatThrownBy(() -> EnhanceConfig.builder().build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
		assertThatThrownBy(() -> EnhanceConfig.builder().modelPath("").build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}

	@Test
	void rejectsEmptyModelVersion() {
		EnhanceConfig config = EnhanceConfig.builder()
			.modelPath(MODEL).modelVersion("").build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelVersion");
	}

	@Test
	void rejectsNonPositiveScale() {
		EnhanceConfig config = EnhanceConfig.builder().modelPath(MODEL).scale(0).build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("scale");
	}

	@Test
	void rejectsNonPositiveTileSize() {
		EnhanceConfig config = EnhanceConfig.builder().modelPath(MODEL).tileSize(0).build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("tileSize");
	}

	@Test
	void rejectsNegativeOverlap() {
		EnhanceConfig config = EnhanceConfig.builder().modelPath(MODEL).tileOverlap(-1).build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("tileOverlap");
	}

	@Test
	void rejectsOverlapNotSmallerThanTile() {
		EnhanceConfig config = EnhanceConfig.builder()
			.modelPath(MODEL)
			.tileSize(32)
			.tileOverlap(32)
			.build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("tileOverlap");
	}
}
