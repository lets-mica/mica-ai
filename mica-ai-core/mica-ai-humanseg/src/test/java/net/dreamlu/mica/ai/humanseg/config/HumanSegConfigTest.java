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
package net.dreamlu.mica.ai.humanseg.config;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link HumanSegConfig} 单元测试。
 */
class HumanSegConfigTest {

	@Test
	void defaultConfigMatchesBundledModel() {
		HumanSegConfig config = HumanSegConfig.builder().build();

		assertThat(config.getModelVersion())
			.isEqualTo("human_segmentation_pphumanseg_2023mar");
		assertThat(config.getInputWidth()).isEqualTo(192);
		assertThat(config.getInputHeight()).isEqualTo(192);
		assertThat(config.getMean()).containsExactly(0.5f, 0.5f, 0.5f);
		assertThat(config.getStd()).containsExactly(0.5f, 0.5f, 0.5f);
		assertThat(config.getBinaryThreshold()).isEqualTo(0.5f);
		assertThat(config.getBackgroundColor()).containsExactly(255, 255, 255);
		config.validate();
	}

	@Test
	void resolveModelPathUsesClasspathResource() {
		assertThat(HumanSegConfig.builder().build().resolveModelPath())
			.isEqualTo("classpath:mica-ai/models/humanseg/"
				+ "human_segmentation_pphumanseg_2023mar.onnx");
	}

	@Test
	void explicitModelPathWins() {
		HumanSegConfig config = HumanSegConfig.builder()
			.modelPath("D:/models/humanseg.onnx")
			.build();

		assertThat(config.resolveModelPath()).isEqualTo("D:/models/humanseg.onnx");
	}

	@Test
	void acceptsPortraitInputSize() {
		HumanSegConfig config = HumanSegConfig.builder()
			.modelVersion("pphumansegv2_lite_portrait")
			.inputWidth(256)
			.inputHeight(144)
			.build();
		config.validate();

		assertThat(config.getInputWidth()).isEqualTo(256);
		assertThat(config.getInputHeight()).isEqualTo(144);
	}

	@Test
	void rejectsEmptyModelVersion() {
		assertThatThrownBy(() -> HumanSegConfig.builder()
			.modelVersion("").build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelVersion");
	}

	@Test
	void rejectsNonPositiveInputSize() {
		assertThatThrownBy(() -> HumanSegConfig.builder().inputWidth(0).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputWidth");
		assertThatThrownBy(() -> HumanSegConfig.builder().inputHeight(-1).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputWidth");
	}

	@Test
	void rejectsMalformedMeanAndStd() {
		assertThatThrownBy(() -> HumanSegConfig.builder()
			.mean(new float[]{0.5f}).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("mean");
		assertThatThrownBy(() -> HumanSegConfig.builder()
			.std(new float[]{0.5f, 0.5f}).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("mean");
	}

	@Test
	void rejectsZeroStd() {
		assertThatThrownBy(() -> HumanSegConfig.builder()
			.std(new float[]{0.5f, 0f, 0.5f}).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("std[1]");
	}

	@Test
	void rejectsBinaryThresholdOutOfRange() {
		assertThatThrownBy(() -> HumanSegConfig.builder()
			.binaryThreshold(1.5f).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("binaryThreshold");
		assertThatThrownBy(() -> HumanSegConfig.builder()
			.binaryThreshold(-0.1f).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("binaryThreshold");
	}

	@Test
	void rejectsMalformedBackgroundColor() {
		assertThatThrownBy(() -> HumanSegConfig.builder()
			.backgroundColor(new int[]{255, 0}).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("backgroundColor");
		assertThatThrownBy(() -> HumanSegConfig.builder()
			.backgroundColor(new int[]{255, 0, 300}).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("backgroundColor[2]");
	}
}
