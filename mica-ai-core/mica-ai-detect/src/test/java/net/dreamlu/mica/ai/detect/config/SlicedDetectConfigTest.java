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
import net.dreamlu.mica.ai.detect.model.SliceMergeStrategy;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link SlicedDetectConfig} 配置与校验测试。
 */
class SlicedDetectConfigTest {

	private static DetectConfig detectConfig() {
		return DetectConfig.builder()
			.modelPath("model-tools/detect/models/object_detection_nanodet_2022nov.onnx")
			.family(DetectModelFamily.NANODET_PLUS)
			.inputSize(416)
			.build();
	}

	private static SlicedDetectConfig config() {
		return SlicedDetectConfig.builder().detectConfig(detectConfig()).build();
	}

	@Test
	void defaultsMatchPlan() {
		SlicedDetectConfig config = config();

		assertThat(config.getOverlapRatio()).isEqualTo(0.25f);
		assertThat(config.getMaxSlices()).isEqualTo(64);
		assertThat(config.getMergeStrategy()).isEqualTo(SliceMergeStrategy.IOU_MERGE);
		config.validate();
	}

	@Test
	void sliceSizeAlwaysFollowsModelInputSize() {
		SlicedDetectConfig config = config();

		// 切片边长不是配置项，恒等于模型输入边长
		assertThat(config.resolveSliceSize()).isEqualTo(416);
		// 步进 = 边长 - 重叠像素 = 416 - round(416 * 0.25) = 416 - 104 = 312
		assertThat(config.resolveStep()).isEqualTo(312);
	}

	@Test
	void stepNeverBecomesZero() {
		SlicedDetectConfig config = SlicedDetectConfig.builder()
			.detectConfig(detectConfig())
			.overlapRatio(0.999f)
			.build();

		assertThat(config.resolveStep()).isEqualTo(1);
	}

	@Test
	void rejectsNullDetectConfig() {
		SlicedDetectConfig config = SlicedDetectConfig.builder().build();

		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("detectConfig");
	}

	@Test
	void rejectsIllegalOverlapRatio() {
		SlicedDetectConfig zero = SlicedDetectConfig.builder()
			.detectConfig(detectConfig()).overlapRatio(0f).build();
		SlicedDetectConfig one = SlicedDetectConfig.builder()
			.detectConfig(detectConfig()).overlapRatio(1f).build();

		assertThatThrownBy(zero::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("overlapRatio");
		assertThatThrownBy(one::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("overlapRatio");
	}

	@Test
	void rejectsIllegalMaxSlices() {
		SlicedDetectConfig config = SlicedDetectConfig.builder()
			.detectConfig(detectConfig()).maxSlices(0).build();

		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("maxSlices");
	}

	@Test
	void propagatesDetectConfigValidation() {
		SlicedDetectConfig config = SlicedDetectConfig.builder()
			.detectConfig(DetectConfig.builder().build())
			.build();

		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}

}
