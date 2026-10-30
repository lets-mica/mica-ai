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
package net.dreamlu.mica.ai.detect.config;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link DetectConfig} 单元测试。
 */
class DetectConfigTest {

	@Test
	void defaultConfigIsNanodetPlus() {
		DetectConfig config = DetectConfig.builder().build();

		assertThat(config.getFamily()).isEqualTo(DetectModelFamily.NANODET_PLUS);
		assertThat(config.getInputSize()).isEqualTo(416);
		assertThat(config.getScoreThreshold()).isEqualTo(0.35f);
		assertThat(config.getNmsThreshold()).isEqualTo(0.6f);
		assertThat(config.getMaxDetections()).isEqualTo(100);
		assertThat(config.getLabels()).isNull();
		config.validate();
	}

	@Test
	void resolveModelPathUsesFamilyResource() {
		assertThat(DetectConfig.builder().build().resolveModelPath())
			.isEqualTo("classpath:mica-ai/models/detect/nanodet_plus.onnx");

		assertThat(DetectConfig.builder().family(DetectModelFamily.YOLOX_S).build().resolveModelPath())
			.isEqualTo("classpath:mica-ai/models/detect/yolox_s.onnx");
	}

	@Test
	void explicitModelPathWins() {
		DetectConfig config = DetectConfig.builder()
			.modelPath("D:/models/nanodet.onnx")
			.build();

		assertThat(config.resolveModelPath()).isEqualTo("D:/models/nanodet.onnx");
	}

	@Test
	void familyCarriesNativeInputSize() {
		assertThat(DetectModelFamily.NANODET_PLUS.getInputSize()).isEqualTo(416);
		assertThat(DetectModelFamily.YOLOX_S.getInputSize()).isEqualTo(640);
	}

	@Test
	void rejectsNullFamily() {
		assertThatThrownBy(() -> DetectConfig.builder().family(null).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("family");
	}

	@Test
	void rejectsNonPositiveInputSize() {
		assertThatThrownBy(() -> DetectConfig.builder().inputSize(0).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputSize");
	}

	@Test
	void rejectsInputSizeNotMultipleOf32() {
		assertThatThrownBy(() -> DetectConfig.builder().inputSize(410).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("32");
	}

	@Test
	void acceptsYoloxInputSize() {
		DetectConfig config = DetectConfig.builder()
			.family(DetectModelFamily.YOLOX_S)
			.inputSize(640)
			.scoreThreshold(0.5f)
			.nmsThreshold(0.5f)
			.build();
		config.validate();
		assertThat(config.getInputSize()).isEqualTo(640);
	}

	@Test
	void rejectsScoreThresholdOutOfRange() {
		assertThatThrownBy(() -> DetectConfig.builder().scoreThreshold(1.5f).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("scoreThreshold");
		assertThatThrownBy(() -> DetectConfig.builder().scoreThreshold(-0.1f).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("scoreThreshold");
	}

	@Test
	void rejectsNmsThresholdOutOfRange() {
		assertThatThrownBy(() -> DetectConfig.builder().nmsThreshold(1.2f).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("nmsThreshold");
	}

	@Test
	void rejectsNonPositiveMaxDetections() {
		assertThatThrownBy(() -> DetectConfig.builder().maxDetections(0).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("maxDetections");
	}

	@Test
	void rejectsEmptyLabelsArray() {
		assertThatThrownBy(() -> DetectConfig.builder()
			.labels(new String[0]).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("labels");
	}
}
