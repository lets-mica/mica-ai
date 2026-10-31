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
package net.dreamlu.mica.ai.classify.config;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link ClassifyConfig} 单元测试。
 */
class ClassifyConfigTest {

	private static final String MODEL = "model-tools/classify/models/text_image_orientation.onnx";
	private static final String LABELS = "model-tools/classify/models/text_image_orientation_labels.txt";

	@Test
	void defaultsMatchPulcContract() {
		ClassifyConfig config = ClassifyConfig.builder()
			.modelPath(MODEL)
			.labelsPath(LABELS)
			.build();
		config.validate();

		assertThat(config.getModelVersion()).isEqualTo("text_image_orientation");
		assertThat(config.getInputWidth()).isEqualTo(224);
		assertThat(config.getInputHeight()).isEqualTo(224);
		assertThat(config.getResizeShort()).isEqualTo(256);
		assertThat(config.getChannelOrder()).isEqualTo(ClassifyChannelOrder.RGB);
		assertThat(config.getMean()).containsExactly(0.485f, 0.456f, 0.406f);
		assertThat(config.getStd()).containsExactly(0.229f, 0.224f, 0.225f);
		assertThat(config.getInterpolation()).isEqualTo(ClassifyInterpolation.LINEAR);
		assertThat(config.isMultiLabel()).isFalse();
		assertThat(config.isOutputIsProbability()).isTrue();
		assertThat(config.getTopK()).isEqualTo(1);
	}

	@Test
	void resolveModelPathKeepsExplicitClasspathPrefix() {
		ClassifyConfig config = ClassifyConfig.builder()
			.modelPath("classpath:models/classify.onnx")
			.labelsPath(LABELS)
			.build();

		assertThat(config.resolveModelPath()).isEqualTo("classpath:models/classify.onnx");
	}

	@Test
	void explicitModelPathWins() {
		ClassifyConfig config = ClassifyConfig.builder()
			.labelsPath(LABELS)
			.modelPath("D:/models/classify.onnx")
			.build();

		assertThat(config.resolveModelPath()).isEqualTo("D:/models/classify.onnx");
	}

	@Test
	void acceptsTableAttributeStyleMultiLabelConfig() {
		ClassifyConfig config = ClassifyConfig.builder()
			.modelPath("model-tools/classify/models/table_attribute.onnx")
			.modelVersion("table_attribute")
			.labelsPath("model-tools/classify/models/table_attribute_labels.txt")
			.resizeShort(0)
			.multiLabel(true)
			.topK(6)
			.build();
		config.validate();

		assertThat(config.getResizeShort()).isZero();
		assertThat(config.isMultiLabel()).isTrue();
		assertThat(config.getTopK()).isEqualTo(6);
	}

	@Test
	void rejectsMissingModelPath() {
		assertThatThrownBy(() -> ClassifyConfig.builder().labelsPath(LABELS).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
		assertThatThrownBy(() -> ClassifyConfig.builder().modelPath("").labelsPath(LABELS).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}

	@Test
	void rejectsMissingLabelsPath() {
		assertThatThrownBy(() -> ClassifyConfig.builder().modelPath(MODEL).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("labelsPath");
		assertThatThrownBy(() -> ClassifyConfig.builder().modelPath(MODEL).labelsPath("").build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("labelsPath");
	}

	@Test
	void rejectsEmptyModelVersion() {
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).modelVersion("").labelsPath(LABELS).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelVersion");
	}

	@Test
	void rejectsNonPositiveInputSize() {
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).inputWidth(0).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputWidth");
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).inputHeight(-1).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputWidth");
	}

	@Test
	void rejectsNegativeResizeShort() {
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).resizeShort(-1).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("resizeShort");
	}

	@Test
	void rejectsResizeShortSmallerThanInputSize() {
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).resizeShort(200).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("resizeShort");
	}

	@Test
	void rejectsMalformedMeanAndStd() {
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).mean(new float[]{0.5f}).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("mean");
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).std(new float[]{0.5f, 0.5f}).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("mean");
	}

	@Test
	void rejectsZeroStd() {
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).std(new float[]{0.5f, 0f, 0.5f}).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("std[1]");
	}

	@Test
	void rejectsNullChannelOrderAndInterpolation() {
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).channelOrder(null).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("channelOrder");
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).interpolation(null).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("interpolation");
	}

	@Test
	void rejectsNonPositiveTopK() {
		assertThatThrownBy(() -> ClassifyConfig.builder()
			.modelPath(MODEL).labelsPath(LABELS).topK(0).build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("topK");
	}
}
