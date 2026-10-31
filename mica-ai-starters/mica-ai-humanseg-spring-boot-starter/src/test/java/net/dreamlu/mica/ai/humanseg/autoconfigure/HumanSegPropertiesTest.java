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
package net.dreamlu.mica.ai.humanseg.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.humanseg.config.HumanSegConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link HumanSegProperties} 与 {@link HumanSegAutoConfiguration#toConfig} 映射测试。
 */
class HumanSegPropertiesTest {

	@Test
	void defaultsMatchBundledModel() {
		HumanSegProperties properties = new HumanSegProperties();
		HumanSegConfig config = HumanSegAutoConfiguration.toConfig(properties);

		assertThat(properties.isEnabled()).isTrue();
		assertThat(config.getModelVersion())
			.isEqualTo("human_segmentation_pphumanseg_2023mar");
		assertThat(config.getInputWidth()).isEqualTo(192);
		assertThat(config.getInputHeight()).isEqualTo(192);
		assertThat(config.getMean()).containsExactly(0.5f, 0.5f, 0.5f);
		assertThat(config.getStd()).containsExactly(0.5f, 0.5f, 0.5f);
		assertThat(config.getBinaryThreshold()).isEqualTo(0.5f);
		assertThat(config.getBackgroundColor()).containsExactly(255, 255, 255);
		assertThat(config.getModelPath()).isNull();
		// 模型不随 jar 分发，没有内置 classpath 兜底：未配置模型路径必须启动即失败
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}

	@Test
	void mapsAllFieldsToCoreConfig() {
		HumanSegProperties properties = new HumanSegProperties();
		properties.setModelPath("D:/models/humanseg.onnx");
		properties.setModelVersion("pphumansegv2_lite_portrait");
		properties.setInputWidth(256);
		properties.setInputHeight(144);
		properties.setMean(new float[]{0.2f, 0.3f, 0.4f});
		properties.setStd(new float[]{0.1f, 0.2f, 0.3f});
		properties.setBinaryThreshold(0.6f);
		properties.setBackgroundColor(new int[]{10, 20, 30});

		HumanSegConfig config = HumanSegAutoConfiguration.toConfig(properties);

		assertThat(config.getModelPath()).isEqualTo("D:/models/humanseg.onnx");
		assertThat(config.resolveModelPath()).isEqualTo("D:/models/humanseg.onnx");
		assertThat(config.getModelVersion()).isEqualTo("pphumansegv2_lite_portrait");
		assertThat(config.getInputWidth()).isEqualTo(256);
		assertThat(config.getInputHeight()).isEqualTo(144);
		assertThat(config.getMean()).containsExactly(0.2f, 0.3f, 0.4f);
		assertThat(config.getStd()).containsExactly(0.1f, 0.2f, 0.3f);
		assertThat(config.getBinaryThreshold()).isEqualTo(0.6f);
		assertThat(config.getBackgroundColor()).containsExactly(10, 20, 30);
		config.validate();
	}

	@Test
	void sharesOnnxOptionsInstance() {
		HumanSegProperties properties = new HumanSegProperties();
		HumanSegConfig config = HumanSegAutoConfiguration.toConfig(properties);

		assertThat(config.getOnnx()).isSameAs(properties.getOnnx());
	}
}
