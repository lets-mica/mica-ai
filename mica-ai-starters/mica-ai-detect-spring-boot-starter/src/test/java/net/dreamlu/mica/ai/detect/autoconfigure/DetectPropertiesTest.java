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
package net.dreamlu.mica.ai.detect.autoconfigure;

import net.dreamlu.mica.ai.detect.config.DetectConfig;
import net.dreamlu.mica.ai.detect.config.DetectModelFamily;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link DetectProperties} 与 {@link DetectAutoConfiguration#toConfig} 映射测试。
 */
class DetectPropertiesTest {

	@Test
	void defaultsMatchNanodetPlus() {
		DetectProperties properties = new DetectProperties();
		DetectConfig config = DetectAutoConfiguration.toConfig(properties);

		assertThat(properties.isEnabled()).isTrue();
		assertThat(config.getFamily()).isEqualTo(DetectModelFamily.NANODET_PLUS);
		assertThat(config.getInputSize()).isEqualTo(416);
		assertThat(config.getScoreThreshold()).isEqualTo(0.35f);
		assertThat(config.getNmsThreshold()).isEqualTo(0.6f);
		assertThat(config.getMaxDetections()).isEqualTo(100);
		assertThat(config.getLabels()).isNull();
		config.validate();
	}

	@Test
	void mapsAllFieldsToCoreConfig() {
		DetectProperties properties = new DetectProperties();
		properties.setModelPath("D:/models/yolox.onnx");
		properties.setFamily(DetectModelFamily.YOLOX_S);
		properties.setInputSize(640);
		properties.setScoreThreshold(0.5f);
		properties.setNmsThreshold(0.45f);
		properties.setMaxDetections(50);
		properties.setLabels(new String[]{"person", "car"});

		DetectConfig config = DetectAutoConfiguration.toConfig(properties);

		assertThat(config.getModelPath()).isEqualTo("D:/models/yolox.onnx");
		assertThat(config.resolveModelPath()).isEqualTo("D:/models/yolox.onnx");
		assertThat(config.getFamily()).isEqualTo(DetectModelFamily.YOLOX_S);
		assertThat(config.getInputSize()).isEqualTo(640);
		assertThat(config.getScoreThreshold()).isEqualTo(0.5f);
		assertThat(config.getNmsThreshold()).isEqualTo(0.45f);
		assertThat(config.getMaxDetections()).isEqualTo(50);
		assertThat(config.getLabels()).containsExactly("person", "car");
		config.validate();
	}

	@Test
	void sharesOnnxOptionsInstance() {
		DetectProperties properties = new DetectProperties();
		DetectConfig config = DetectAutoConfiguration.toConfig(properties);

		assertThat(config.getOnnx()).isSameAs(properties.getOnnx());
	}
}
