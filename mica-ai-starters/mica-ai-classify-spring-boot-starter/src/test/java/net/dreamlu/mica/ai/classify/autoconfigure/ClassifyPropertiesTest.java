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
package net.dreamlu.mica.ai.classify.autoconfigure;

import net.dreamlu.mica.ai.classify.config.ClassifyChannelOrder;
import net.dreamlu.mica.ai.classify.config.ClassifyConfig;
import net.dreamlu.mica.ai.classify.config.ClassifyInterpolation;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link ClassifyProperties} 与 {@link ClassifyAutoConfiguration#toConfig} 映射测试。
 */
class ClassifyPropertiesTest {

	@Test
	void defaultsMatchPulcContract() {
		ClassifyProperties properties = new ClassifyProperties();
		ClassifyConfig config = ClassifyAutoConfiguration.toConfig(properties);

		assertThat(properties.isEnabled()).isTrue();
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
		assertThat(config.getModelPath()).isNull();
		assertThat(config.getLabelsPath()).isNull();
	}

	@Test
	void mapsAllFieldsToCoreConfig() {
		ClassifyProperties properties = new ClassifyProperties();
		properties.setModelPath("D:/models/classify.onnx");
		properties.setModelVersion("table_attribute");
		properties.setLabelsPath("D:/models/table_attribute_labels.txt");
		properties.setInputWidth(224);
		properties.setInputHeight(224);
		properties.setResizeShort(0);
		properties.setChannelOrder(ClassifyChannelOrder.BGR);
		properties.setMean(new float[]{0.1f, 0.2f, 0.3f});
		properties.setStd(new float[]{0.4f, 0.5f, 0.6f});
		properties.setInterpolation(ClassifyInterpolation.CUBIC);
		properties.setMultiLabel(true);
		properties.setOutputIsProbability(false);
		properties.setTopK(6);

		ClassifyConfig config = ClassifyAutoConfiguration.toConfig(properties);

		assertThat(config.getModelPath()).isEqualTo("D:/models/classify.onnx");
		assertThat(config.resolveModelPath()).isEqualTo("D:/models/classify.onnx");
		assertThat(config.getModelVersion()).isEqualTo("table_attribute");
		assertThat(config.getLabelsPath()).isEqualTo("D:/models/table_attribute_labels.txt");
		assertThat(config.getInputWidth()).isEqualTo(224);
		assertThat(config.getInputHeight()).isEqualTo(224);
		assertThat(config.getResizeShort()).isZero();
		assertThat(config.getChannelOrder()).isEqualTo(ClassifyChannelOrder.BGR);
		assertThat(config.getMean()).containsExactly(0.1f, 0.2f, 0.3f);
		assertThat(config.getStd()).containsExactly(0.4f, 0.5f, 0.6f);
		assertThat(config.getInterpolation()).isEqualTo(ClassifyInterpolation.CUBIC);
		assertThat(config.isMultiLabel()).isTrue();
		assertThat(config.isOutputIsProbability()).isFalse();
		assertThat(config.getTopK()).isEqualTo(6);
		config.validate();
	}

	@Test
	void labelsPathIsRequired() {
		ClassifyProperties properties = new ClassifyProperties();
		properties.setModelPath("D:/models/classify.onnx");
		ClassifyConfig config = ClassifyAutoConfiguration.toConfig(properties);

		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("labelsPath");
	}

	@Test
	void modelPathIsRequired() {
		// 模型不随 jar 分发，没有内置 classpath 兜底：未配置模型路径必须启动即失败
		ClassifyConfig config = ClassifyAutoConfiguration.toConfig(new ClassifyProperties());

		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}

	@Test
	void sharesOnnxOptionsInstance() {
		ClassifyProperties properties = new ClassifyProperties();
		ClassifyConfig config = ClassifyAutoConfiguration.toConfig(properties);

		assertThat(config.getOnnx()).isSameAs(properties.getOnnx());
	}
}
