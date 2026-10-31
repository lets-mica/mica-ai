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
package net.dreamlu.mica.ai.filetype.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import net.dreamlu.mica.ai.filetype.config.FiletypeConfig;
import net.dreamlu.mica.ai.filetype.config.PredictionMode;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link FiletypeProperties} 配置绑定测试（纯绑定，不启动 Spring 上下文、不加载模型）。
 *
 * <p>验证目标：文档里写的 {@code mica.ai.filetype.*} 键名真的能绑上、
 * 默认值与 {@link FiletypeConfig} 保持一致、三个资源路径都被装配。
 * 这些是最容易「文档与代码不一致」的地方（改字段名忘了改 yml 即静默失效）。
 */
class FiletypePropertiesTest {

	private static FiletypeProperties bind(Map<String, Object> source) {
		Binder binder = new Binder(new MapConfigurationPropertySource(source));
		return binder.bind("mica.ai.filetype", Bindable.of(FiletypeProperties.class))
			.orElseGet(FiletypeProperties::new);
	}

	@Test
	void defaultsShouldMatchCoreConfigDefaults() {
		FiletypeProperties p = new FiletypeProperties();
		FiletypeConfig c = FiletypeConfig.builder().build();

		assertThat(p.isEnabled()).as("默认应启用").isTrue();
		assertThat(p.getModelVersion()).isEqualTo(c.getModelVersion()).isEqualTo("standard_v3_3");
		assertThat(p.getPredictionMode()).isEqualTo(c.getPredictionMode())
			.isEqualTo(PredictionMode.HIGH_CONFIDENCE);
		assertThat(p.getOnnx().getDevice()).as("默认走 CPU").isEqualTo(OrtDevice.CPU);
	}

	@Test
	void kebabCaseKeysShouldBind() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.filetype.enabled", "false");
		src.put("mica.ai.filetype.model-version", "custom-v1");
		src.put("mica.ai.filetype.model-path", "model-tools/filetype/models/model.onnx");
		src.put("mica.ai.filetype.config-path", "model-tools/filetype/models/config.min.json");
		src.put("mica.ai.filetype.content-types-path",
			"model-tools/filetype/models/content_types_kb.min.json");
		src.put("mica.ai.filetype.prediction-mode", "BEST_GUESS");
		src.put("mica.ai.filetype.onnx.device", "gpu");
		src.put("mica.ai.filetype.onnx.intra-op-num-threads", "4");

		FiletypeProperties p = bind(src);

		assertThat(p.isEnabled()).isFalse();
		assertThat(p.getModelVersion()).isEqualTo("custom-v1");
		assertThat(p.getModelPath()).isEqualTo("model-tools/filetype/models/model.onnx");
		assertThat(p.getConfigPath()).isEqualTo("model-tools/filetype/models/config.min.json");
		assertThat(p.getContentTypesPath())
			.isEqualTo("model-tools/filetype/models/content_types_kb.min.json");
		assertThat(p.getPredictionMode()).isEqualTo(PredictionMode.BEST_GUESS);
		assertThat(p.getOnnx().getDevice()).isEqualTo(OrtDevice.GPU);
		assertThat(p.getOnnx().getIntraOpNumThreads()).isEqualTo(4);
	}

	/**
	 * 装配一致性：{@link FiletypeAutoConfiguration#toConfig} 必须把每个属性都装进
	 * {@link FiletypeConfig}。
	 *
	 * <p>调用的就是自动装配真实使用的那段代码，<b>不在测试里重抄 builder 链</b>——
	 * 重抄的写法无法发现「漏写一个 {@code .xxx(properties.getXxx())}」，形同虚设。
	 */
	@Test
	void autoConfigurationBuilderShouldWireEveryProperty() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.filetype.model-version", "custom-v1");
		src.put("mica.ai.filetype.model-path", "/tmp/model.onnx");
		src.put("mica.ai.filetype.config-path", "/tmp/config.min.json");
		src.put("mica.ai.filetype.content-types-path", "/tmp/kb.min.json");
		src.put("mica.ai.filetype.prediction-mode", "MEDIUM_CONFIDENCE");
		src.put("mica.ai.filetype.onnx.device", "gpu");
		src.put("mica.ai.filetype.onnx.intra-op-num-threads", "2");

		FiletypeProperties p = bind(src);
		FiletypeConfig c = FiletypeAutoConfiguration.toConfig(p);

		assertThat(c.getModelVersion()).isEqualTo("custom-v1");
		assertThat(c.getModelPath()).isEqualTo("/tmp/model.onnx");
		assertThat(c.getConfigPath()).isEqualTo("/tmp/config.min.json");
		assertThat(c.getContentTypesPath()).isEqualTo("/tmp/kb.min.json");
		assertThat(c.getPredictionMode()).isEqualTo(PredictionMode.MEDIUM_CONFIDENCE);
		assertThat(c.getOnnx()).isSameAs(p.getOnnx());
		c.validate();
	}

	/**
	 * 模型与知识库成对使用：三个路径缺一不可，缺哪个就报哪个的名字。
	 */
	@Test
	void unsetResourcePathsShouldFailFast() {
		// 模型不随 jar 分发，没有内置 classpath 兜底：未配置模型路径必须启动即失败
		FiletypeConfig empty = FiletypeConfig.builder().build();
		assertThatThrownBy(empty::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");

		FiletypeConfig onlyModel = FiletypeConfig.builder().modelPath("/tmp/model.onnx").build();
		assertThatThrownBy(onlyModel::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("configPath");

		FiletypeConfig modelAndConfig = FiletypeConfig.builder()
			.modelPath("/tmp/model.onnx")
			.configPath("/tmp/config.min.json")
			.build();
		assertThatThrownBy(modelAndConfig::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("contentTypesPath");
	}
}
