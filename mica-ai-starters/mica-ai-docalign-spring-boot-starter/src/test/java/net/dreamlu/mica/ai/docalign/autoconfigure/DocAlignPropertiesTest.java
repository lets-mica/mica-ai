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
package net.dreamlu.mica.ai.docalign.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.docalign.config.DocAlignConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link DocAlignProperties} 与 {@link DocAlignAutoConfiguration#toConfig} 映射测试。
 */
class DocAlignPropertiesTest {

	@Test
	void defaultsMatchBundledModel() {
		DocAlignProperties properties = new DocAlignProperties();
		DocAlignConfig config = DocAlignAutoConfiguration.toConfig(properties);

		assertThat(properties.isEnabled()).isTrue();
		assertThat(config.getModelVersion()).isEqualTo("lcnet100_h_e_bifpn_256_fp32");
		assertThat(config.getInputWidth()).isEqualTo(256);
		assertThat(config.getInputHeight()).isEqualTo(256);
		assertThat(config.getHeatmapThreshold()).isEqualTo(0.3f);
		assertThat(config.getModelPath()).isNull();
		// 模型不随 jar 分发，没有内置 classpath 兜底：未配置模型路径必须启动即失败
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}

	@Test
	void mapsAllFieldsToCoreConfig() {
		DocAlignProperties properties = new DocAlignProperties();
		properties.setModelPath("D:/models/docalign.onnx");
		properties.setModelVersion("fastvit_t8_h_e_bifpn_256_fp32");
		properties.setInputWidth(256);
		properties.setInputHeight(256);
		properties.setHeatmapThreshold(0.4f);

		DocAlignConfig config = DocAlignAutoConfiguration.toConfig(properties);

		assertThat(config.getModelPath()).isEqualTo("D:/models/docalign.onnx");
		assertThat(config.resolveModelPath()).isEqualTo("D:/models/docalign.onnx");
		assertThat(config.getModelVersion()).isEqualTo("fastvit_t8_h_e_bifpn_256_fp32");
		assertThat(config.getHeatmapThreshold()).isEqualTo(0.4f);
		config.validate();
	}

	@Test
	void sharesOnnxOptionsInstance() {
		DocAlignProperties properties = new DocAlignProperties();
		DocAlignConfig config = DocAlignAutoConfiguration.toConfig(properties);

		assertThat(config.getOnnx()).isSameAs(properties.getOnnx());
	}
}
