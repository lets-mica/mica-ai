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
package net.dreamlu.mica.ai.docalign.config;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link DocAlignConfig} 单元测试。
 */
class DocAlignConfigTest {

	private static final String MODEL =
		"model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx";

	@Test
	void defaultConfigMatchesBundledModel() {
		DocAlignConfig config = DocAlignConfig.builder().modelPath(MODEL).build();

		assertThat(config.getModelVersion()).isEqualTo("lcnet100_h_e_bifpn_256_fp32");
		assertThat(config.getInputWidth()).isEqualTo(256);
		assertThat(config.getInputHeight()).isEqualTo(256);
		assertThat(config.getHeatmapThreshold()).isEqualTo(0.3f);
		config.validate();
	}

	@Test
	void resolveModelPathKeepsExplicitClasspathPrefix() {
		DocAlignConfig config = DocAlignConfig.builder()
			.modelPath("classpath:models/docalign/lcnet100.onnx")
			.build();

		assertThat(config.resolveModelPath()).isEqualTo("classpath:models/docalign/lcnet100.onnx");
	}

	@Test
	void explicitModelPathWins() {
		DocAlignConfig config = DocAlignConfig.builder()
			.modelPath("D:/models/docalign.onnx")
			.build();

		assertThat(config.resolveModelPath()).isEqualTo("D:/models/docalign.onnx");
	}

	@Test
	void rejectsMissingModelPath() {
		assertThatThrownBy(() -> DocAlignConfig.builder().build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
		assertThatThrownBy(() -> DocAlignConfig.builder().modelPath("").build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}

	@Test
	void rejectsEmptyModelVersion() {
		assertThatThrownBy(() -> DocAlignConfig.builder()
			.modelPath(MODEL).modelVersion("").build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelVersion");
	}

	@Test
	void rejectsNonPositiveInputSize() {
		assertThatThrownBy(() -> DocAlignConfig.builder()
			.modelPath(MODEL).inputWidth(0).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputWidth");
		assertThatThrownBy(() -> DocAlignConfig.builder()
			.modelPath(MODEL).inputHeight(-1).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputWidth");
	}

	@Test
	void rejectsHeatmapThresholdOutOfRange() {
		assertThatThrownBy(() -> DocAlignConfig.builder()
			.modelPath(MODEL).heatmapThreshold(1.2f).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("heatmapThreshold");
		assertThatThrownBy(() -> DocAlignConfig.builder()
			.modelPath(MODEL).heatmapThreshold(-0.1f).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("heatmapThreshold");
	}
}
