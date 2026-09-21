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
package net.dreamlu.mica.ai.matting.autoconfigure;

import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import net.dreamlu.mica.ai.matting.config.MattingConfig;
import net.dreamlu.mica.ai.matting.config.MattingInterpolation;
import net.dreamlu.mica.ai.matting.config.MattingOutputSelect;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link MattingProperties} 配置绑定测试（纯绑定，不启动 Spring 上下文、不加载模型）。
 *
 * <p>验证目标：文档里写的 {@code mica.ai.matting.*} 键名真的能绑上、
 * 默认值与 {@link MattingConfig} 保持一致、嵌套 {@code onnx} 能绑到 {@link OrtDevice}。
 * 这些是最容易「文档与代码不一致」的地方（改字段名忘了改 yml 即静默失效）。
 */
class MattingPropertiesTest {

	private static MattingProperties bind(Map<String, Object> source) {
		Binder binder = new Binder(new MapConfigurationPropertySource(source));
		return binder.bind("mica.ai.matting", Bindable.of(MattingProperties.class))
			.orElseGet(MattingProperties::new);
	}

	@Test
	void defaultsShouldMatchCoreConfigDefaults() {
		MattingProperties p = new MattingProperties();
		MattingConfig c = MattingConfig.builder().build();

		assertThat(p.isEnabled()).as("默认应启用").isTrue();
		assertThat(p.getModelVersion()).isEqualTo(c.getModelVersion()).isEqualTo("u2netp");
		assertThat(p.getInputSize()).isEqualTo(c.getInputSize()).isEqualTo(320);
		assertThat(p.getOutputSelect()).isEqualTo(c.getOutputSelect())
			.isEqualTo(MattingOutputSelect.AUTO);
		assertThat(p.getMean()).isEqualTo(c.getMean());
		assertThat(p.getStd()).isEqualTo(c.getStd());
		assertThat(p.getInterpolation()).isEqualTo(c.getInterpolation())
			.isEqualTo(MattingInterpolation.LINEAR);
		assertThat(p.getBinaryThreshold()).isEqualTo(c.getBinaryThreshold()).isEqualTo(0.5f);
		assertThat(p.isMinMaxNormalize()).isEqualTo(c.isMinMaxNormalize()).isTrue();
		assertThat(p.getBackgroundColor()).isEqualTo(c.getBackgroundColor());
		assertThat(p.getOnnx().getDevice()).as("默认走 CPU").isEqualTo(OrtDevice.CPU);
	}

	@Test
	void kebabCaseKeysShouldBind() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.matting.enabled", "false");
		src.put("mica.ai.matting.model-version", "custom-v1");
		src.put("mica.ai.matting.model-path", "model-tools/matting/models/u2netp.onnx");
		src.put("mica.ai.matting.input-size", "320");
		src.put("mica.ai.matting.output-select", "FIRST");
		src.put("mica.ai.matting.binary-threshold", "0.35");
		src.put("mica.ai.matting.min-max-normalize", "false");
		src.put("mica.ai.matting.interpolation", "NEAREST");
		src.put("mica.ai.matting.background-color[0]", "0");
		src.put("mica.ai.matting.background-color[1]", "128");
		src.put("mica.ai.matting.background-color[2]", "255");
		src.put("mica.ai.matting.mean[0]", "0.5");
		src.put("mica.ai.matting.mean[1]", "0.5");
		src.put("mica.ai.matting.mean[2]", "0.5");
		src.put("mica.ai.matting.onnx.device", "gpu");
		src.put("mica.ai.matting.onnx.intra-op-num-threads", "4");

		MattingProperties p = bind(src);

		assertThat(p.isEnabled()).isFalse();
		assertThat(p.getModelVersion()).isEqualTo("custom-v1");
		assertThat(p.getModelPath()).isEqualTo("model-tools/matting/models/u2netp.onnx");
		assertThat(p.getInputSize()).isEqualTo(320);
		assertThat(p.getOutputSelect()).isEqualTo(MattingOutputSelect.FIRST);
		assertThat(p.getBinaryThreshold()).isEqualTo(0.35f);
		assertThat(p.isMinMaxNormalize()).isFalse();
		assertThat(p.getInterpolation()).isEqualTo(MattingInterpolation.NEAREST);
		assertThat(p.getBackgroundColor()).containsExactly(0, 128, 255);
		assertThat(p.getMean()).containsExactly(0.5f, 0.5f, 0.5f);
		assertThat(p.getOnnx().getDevice()).isEqualTo(OrtDevice.GPU);
		assertThat(p.getOnnx().getIntraOpNumThreads()).isEqualTo(4);
	}

	/**
	 * 配置字段最终必须能装进 {@link MattingConfig} 且通过 {@code validate()}，
	 * 否则 starter 里少写一个 {@code .xxx(properties.getXxx())} 会在启动期炸、或更糟：静默用默认值。
	 */
	@Test
	void propertiesShouldBuildValidCoreConfig() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.matting.model-path", "model-tools/matting/models/u2netp.onnx");
		src.put("mica.ai.matting.interpolation", "CUBIC");
		src.put("mica.ai.matting.output-select", "FIRST");
		src.put("mica.ai.matting.binary-threshold", "0.25");
		src.put("mica.ai.matting.background-color[0]", "10");
		src.put("mica.ai.matting.background-color[1]", "20");
		src.put("mica.ai.matting.background-color[2]", "30");

		MattingProperties p = bind(src);
		MattingConfig config = MattingConfig.builder()
			.modelVersion(p.getModelVersion())
			.modelPath(p.getModelPath())
			.inputSize(p.getInputSize())
			.outputSelect(p.getOutputSelect())
			.mean(p.getMean())
			.std(p.getStd())
			.interpolation(p.getInterpolation())
			.binaryThreshold(p.getBinaryThreshold())
			.minMaxNormalize(p.isMinMaxNormalize())
			.backgroundColor(p.getBackgroundColor())
			.onnx(p.getOnnx())
			.build();

		config.validate();
		assertThat(config.resolveModelPath()).isEqualTo("model-tools/matting/models/u2netp.onnx");
		assertThat(config.getInterpolation()).isEqualTo(MattingInterpolation.CUBIC);
		assertThat(config.getOutputSelect()).isEqualTo(MattingOutputSelect.FIRST);
		assertThat(config.getBinaryThreshold()).isEqualTo(0.25f);
		assertThat(config.getBackgroundColor()).containsExactly(10, 20, 30);
	}

	/**
	 * 装配一致性：{@link MattingAutoConfiguration#toConfig} 必须把每个属性都装进
	 * {@link MattingConfig}。
	 *
	 * <p>调用的就是自动装配真实使用的那段代码（{@code mattingEngine} 内联调它），
	 * <b>不在测试里重抄 builder 链</b>——重抄的写法无法发现「漏写一个
	 * {@code .xxx(properties.getXxx())}」，形同虚设。
	 *
	 * <p>传入非默认值绑定，若字段未被装配，核心配置仍是默认值 ⇒ 断言失败。
	 */
	@Test
	void autoConfigurationBuilderShouldWireEveryProperty() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.matting.model-path", "/tmp/external.onnx");
		src.put("mica.ai.matting.model-version", "u2net_human_seg");
		src.put("mica.ai.matting.input-size", "320");
		src.put("mica.ai.matting.output-select", "FIRST");
		src.put("mica.ai.matting.mean[0]", "0.5");
		src.put("mica.ai.matting.mean[1]", "0.5");
		src.put("mica.ai.matting.mean[2]", "0.5");
		src.put("mica.ai.matting.std[0]", "0.25");
		src.put("mica.ai.matting.std[1]", "0.25");
		src.put("mica.ai.matting.std[2]", "0.25");
		src.put("mica.ai.matting.interpolation", "NEAREST");
		src.put("mica.ai.matting.binary-threshold", "0.7");
		src.put("mica.ai.matting.min-max-normalize", "false");
		src.put("mica.ai.matting.background-color[0]", "1");
		src.put("mica.ai.matting.background-color[1]", "2");
		src.put("mica.ai.matting.background-color[2]", "3");

		MattingProperties p = bind(src);
		MattingConfig c = MattingAutoConfiguration.toConfig(p);

		assertThat(c.getModelVersion()).isEqualTo("u2net_human_seg");
		assertThat(c.getModelPath()).isEqualTo("/tmp/external.onnx");
		assertThat(c.getInputSize()).isEqualTo(320);
		assertThat(c.getOutputSelect()).as("output-select 必须被装配")
			.isEqualTo(MattingOutputSelect.FIRST);
		assertThat(c.getMean()).containsExactly(0.5f, 0.5f, 0.5f);
		assertThat(c.getStd()).containsExactly(0.25f, 0.25f, 0.25f);
		assertThat(c.getInterpolation()).isEqualTo(MattingInterpolation.NEAREST);
		assertThat(c.getBinaryThreshold()).isEqualTo(0.7f);
		assertThat(c.isMinMaxNormalize()).isFalse();
		assertThat(c.getBackgroundColor()).containsExactly(1, 2, 3);
	}

	@Test
	void unsetModelPathShouldFallBackToClasspathTemplate() {
		MattingConfig config = MattingConfig.builder().build();
		assertThat(config.resolveModelPath())
			.isEqualTo("classpath:mica-ai/models/matting/u2netp.onnx");
	}
}
