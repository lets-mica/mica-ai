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

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.detect.config.DetectConfig;
import net.dreamlu.mica.ai.detect.config.DetectModelFamily;
import net.dreamlu.mica.ai.detect.config.SlicedDetectConfig;
import net.dreamlu.mica.ai.detect.model.SliceMergeStrategy;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

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
		assertThat(config.getModelPath()).isNull();
		// 模型不随 jar 分发，没有内置 classpath 兜底：未配置模型路径必须启动即失败
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
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

	@Test
	void slicedDefaultsMatchPlan() {
		DetectProperties properties = new DetectProperties();
		DetectProperties.Sliced sliced = properties.getSliced();

		assertThat(sliced.isEnabled()).isTrue();
		assertThat(sliced.getOverlapRatio()).isEqualTo(0.25f);
		assertThat(sliced.getMaxSlices()).isEqualTo(64);
		assertThat(sliced.getMergeStrategy()).isEqualTo(SliceMergeStrategy.IOU_MERGE);
	}

	@Test
	void mapsSlicedPropertiesToCoreConfig() {
		DetectProperties properties = new DetectProperties();
		properties.setModelPath("D:/models/nanodet.onnx");
		properties.getSliced().setOverlapRatio(0.4f);
		properties.getSliced().setMaxSlices(32);
		properties.getSliced().setMergeStrategy(SliceMergeStrategy.KEEP_ALL);

		SlicedDetectConfig config = DetectAutoConfiguration.toSlicedConfig(properties);

		assertThat(config.getOverlapRatio()).isEqualTo(0.4f);
		assertThat(config.getMaxSlices()).isEqualTo(32);
		assertThat(config.getMergeStrategy()).isEqualTo(SliceMergeStrategy.KEEP_ALL);
		// 切片配置委托整图检测配置，两者的模型与阈值口径必须一致
		assertThat(config.getDetectConfig().getModelPath()).isEqualTo("D:/models/nanodet.onnx");
		assertThat(config.getDetectConfig().getFamily()).isEqualTo(DetectModelFamily.NANODET_PLUS);
		// 切片边长恒等于模型输入边长
		assertThat(config.resolveSliceSize()).isEqualTo(416);
		config.validate();
	}
}
