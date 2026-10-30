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
package net.dreamlu.mica.ai.enhance.autoconfigure;

import net.dreamlu.mica.ai.enhance.config.EnhanceConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link EnhanceProperties} 与 {@link EnhanceAutoConfiguration#toConfig} 映射测试。
 */
class EnhancePropertiesTest {

	@Test
	void defaultsMatchBundledModel() {
		EnhanceProperties properties = new EnhanceProperties();
		EnhanceConfig config = EnhanceAutoConfiguration.toConfig(properties);

		assertThat(properties.isEnabled()).isTrue();
		assertThat(config.getModelVersion()).isEqualTo("realesr-general-x4v3");
		assertThat(config.getScale()).isEqualTo(4);
		assertThat(config.getTileSize()).isEqualTo(256);
		assertThat(config.getTileOverlap()).isEqualTo(16);
		assertThat(config.getModelPath()).isNull();
		config.validate();
	}

	@Test
	void mapsAllFieldsToCoreConfig() {
		EnhanceProperties properties = new EnhanceProperties();
		properties.setModelPath("D:/models/realesr.onnx");
		properties.setModelVersion("realesr-general-wdn-x4v3");
		properties.setScale(2);
		properties.setTileSize(512);
		properties.setTileOverlap(32);

		EnhanceConfig config = EnhanceAutoConfiguration.toConfig(properties);

		assertThat(config.getModelPath()).isEqualTo("D:/models/realesr.onnx");
		assertThat(config.resolveModelPath()).isEqualTo("D:/models/realesr.onnx");
		assertThat(config.getModelVersion()).isEqualTo("realesr-general-wdn-x4v3");
		assertThat(config.getScale()).isEqualTo(2);
		assertThat(config.getTileSize()).isEqualTo(512);
		assertThat(config.getTileOverlap()).isEqualTo(32);
		config.validate();
	}

	@Test
	void sharesOnnxOptionsInstance() {
		EnhanceProperties properties = new EnhanceProperties();
		EnhanceConfig config = EnhanceAutoConfiguration.toConfig(properties);

		assertThat(config.getOnnx()).isSameAs(properties.getOnnx());
	}
}
