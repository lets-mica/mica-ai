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
package net.dreamlu.mica.ai.plate.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import net.dreamlu.mica.ai.plate.config.PlateConfig;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link PlateProperties} 配置绑定测试（纯绑定，不启动 Spring 上下文、不加载模型）。
 *
 * <p>验证目标：文档里写的 {@code mica.ai.plate.*} 键名真的能绑上、
 * 默认值与 {@link PlateConfig} 保持一致、三个模型路径都被装配。
 */
class PlatePropertiesTest {

	private static PlateProperties bind(Map<String, Object> source) {
		Binder binder = new Binder(new MapConfigurationPropertySource(source));
		return binder.bind("mica.ai.plate", Bindable.of(PlateProperties.class))
			.orElseGet(PlateProperties::new);
	}

	@Test
	void defaultsShouldMatchCoreConfigDefaults() {
		PlateProperties p = new PlateProperties();
		PlateConfig c = PlateConfig.builder().build();

		assertThat(p.isEnabled()).as("默认应启用").isTrue();
		assertThat(p.getModelVersion()).isEqualTo(c.getModelVersion()).isEqualTo("20230229");
		assertThat(p.getDetectionInputSize()).isEqualTo(c.getDetectionInputSize()).isEqualTo(320);
		assertThat(p.getRecognitionInputHeight()).isEqualTo(c.getRecognitionInputHeight()).isEqualTo(48);
		assertThat(p.getRecognitionInputWidth()).isEqualTo(c.getRecognitionInputWidth()).isEqualTo(160);
		assertThat(p.getClassificationInputSize()).isEqualTo(c.getClassificationInputSize()).isEqualTo(96);
		assertThat(p.getDetectionConfidenceThreshold()).isEqualTo(c.getDetectionConfidenceThreshold())
			.isEqualTo(0.25f);
		assertThat(p.getDetectionNmsThreshold()).isEqualTo(c.getDetectionNmsThreshold()).isEqualTo(0.5f);
		assertThat(p.getMaxPlates()).isEqualTo(c.getMaxPlates()).isEqualTo(5);
		assertThat(p.getOnnx().getDevice()).as("默认走 CPU").isEqualTo(OrtDevice.CPU);
	}

	@Test
	void kebabCaseKeysShouldBind() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.plate.enabled", "false");
		src.put("mica.ai.plate.model-version", "custom-v1");
		src.put("mica.ai.plate.detection-model-path", "model-tools/plate/models/y5fu_640x_sim.onnx");
		src.put("mica.ai.plate.recognition-model-path",
			"model-tools/plate/models/rpv3_mdict_160_r3.onnx");
		src.put("mica.ai.plate.classification-model-path",
			"model-tools/plate/models/litemodel_cls_96x_r1.onnx");
		src.put("mica.ai.plate.detection-input-size", "640");
		src.put("mica.ai.plate.recognition-input-height", "48");
		src.put("mica.ai.plate.recognition-input-width", "160");
		src.put("mica.ai.plate.classification-input-size", "96");
		src.put("mica.ai.plate.detection-confidence-threshold", "0.4");
		src.put("mica.ai.plate.detection-nms-threshold", "0.6");
		src.put("mica.ai.plate.max-plates", "10");
		src.put("mica.ai.plate.onnx.device", "gpu");
		src.put("mica.ai.plate.onnx.intra-op-num-threads", "4");

		PlateProperties p = bind(src);

		assertThat(p.isEnabled()).isFalse();
		assertThat(p.getModelVersion()).isEqualTo("custom-v1");
		assertThat(p.getDetectionModelPath()).isEqualTo("model-tools/plate/models/y5fu_640x_sim.onnx");
		assertThat(p.getRecognitionModelPath())
			.isEqualTo("model-tools/plate/models/rpv3_mdict_160_r3.onnx");
		assertThat(p.getClassificationModelPath())
			.isEqualTo("model-tools/plate/models/litemodel_cls_96x_r1.onnx");
		assertThat(p.getDetectionInputSize()).isEqualTo(640);
		assertThat(p.getRecognitionInputHeight()).isEqualTo(48);
		assertThat(p.getRecognitionInputWidth()).isEqualTo(160);
		assertThat(p.getClassificationInputSize()).isEqualTo(96);
		assertThat(p.getDetectionConfidenceThreshold()).isEqualTo(0.4f);
		assertThat(p.getDetectionNmsThreshold()).isEqualTo(0.6f);
		assertThat(p.getMaxPlates()).isEqualTo(10);
		assertThat(p.getOnnx().getDevice()).isEqualTo(OrtDevice.GPU);
		assertThat(p.getOnnx().getIntraOpNumThreads()).isEqualTo(4);
	}

	/**
	 * 装配一致性：{@link PlateAutoConfiguration#toConfig} 必须把每个属性都装进
	 * {@link PlateConfig}。
	 */
	@Test
	void autoConfigurationBuilderShouldWireEveryProperty() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.plate.model-version", "custom-v1");
		src.put("mica.ai.plate.detection-model-path", "/tmp/det.onnx");
		src.put("mica.ai.plate.recognition-model-path", "/tmp/rec.onnx");
		src.put("mica.ai.plate.classification-model-path", "/tmp/cls.onnx");
		src.put("mica.ai.plate.detection-input-size", "640");
		src.put("mica.ai.plate.recognition-input-height", "48");
		src.put("mica.ai.plate.recognition-input-width", "192");
		src.put("mica.ai.plate.classification-input-size", "96");
		src.put("mica.ai.plate.detection-confidence-threshold", "0.4");
		src.put("mica.ai.plate.detection-nms-threshold", "0.6");
		src.put("mica.ai.plate.max-plates", "10");
		src.put("mica.ai.plate.onnx.device", "gpu");

		PlateProperties p = bind(src);
		PlateConfig c = PlateAutoConfiguration.toConfig(p);

		assertThat(c.getModelVersion()).isEqualTo("custom-v1");
		assertThat(c.getDetectionModelPath()).isEqualTo("/tmp/det.onnx");
		assertThat(c.getRecognitionModelPath()).isEqualTo("/tmp/rec.onnx");
		assertThat(c.getClassificationModelPath()).isEqualTo("/tmp/cls.onnx");
		assertThat(c.getDetectionInputSize()).isEqualTo(640);
		assertThat(c.getRecognitionInputHeight()).isEqualTo(48);
		assertThat(c.getRecognitionInputWidth()).isEqualTo(192);
		assertThat(c.getClassificationInputSize()).isEqualTo(96);
		assertThat(c.getDetectionConfidenceThreshold()).isEqualTo(0.4f);
		assertThat(c.getDetectionNmsThreshold()).isEqualTo(0.6f);
		assertThat(c.getMaxPlates()).isEqualTo(10);
		assertThat(c.getOnnx()).isSameAs(p.getOnnx());
		c.validate();
	}

	/**
	 * 检测模型输入边长只支持 320 / 640，其它值必须快速失败。
	 */
	@Test
	void illegalDetectionInputSizeShouldFailFast() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.plate.detection-input-size", "512");

		PlateConfig c = PlateAutoConfiguration.toConfig(bind(src));
		assertThatThrownBy(c::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("detectionInputSize");
	}
}
