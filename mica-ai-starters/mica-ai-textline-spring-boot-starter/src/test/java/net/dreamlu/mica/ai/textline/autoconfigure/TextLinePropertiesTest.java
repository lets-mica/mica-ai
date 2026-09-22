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
package net.dreamlu.mica.ai.textline.autoconfigure;

import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import net.dreamlu.mica.ai.textline.config.TextLineChannelOrder;
import net.dreamlu.mica.ai.textline.config.TextLineConfig;
import net.dreamlu.mica.ai.textline.config.TextLineInterpolation;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link TextLineProperties} 配置绑定测试（纯绑定，不启动 Spring 上下文、不加载模型）。
 *
 * <p>验证目标：文档里写的 {@code mica.ai.textline.*} 键名真的能绑上、
 * 默认值与 {@link TextLineConfig} 保持一致、嵌套 {@code onnx} 能绑到 {@link OrtDevice}。
 * 这些是最容易「文档与代码不一致」的地方（改字段名忘了改 yml 即静默失效）。
 */
class TextLinePropertiesTest {

	private static TextLineProperties bind(Map<String, Object> source) {
		Binder binder = new Binder(new MapConfigurationPropertySource(source));
		return binder.bind("mica.ai.textline", Bindable.of(TextLineProperties.class))
			.orElseGet(TextLineProperties::new);
	}

	@Test
	void defaultsShouldMatchCoreConfigDefaults() {
		TextLineProperties p = new TextLineProperties();
		TextLineConfig c = TextLineConfig.builder().build();

		assertThat(p.isEnabled()).as("默认应启用").isTrue();
		assertThat(p.getModelVersion()).isEqualTo(c.getModelVersion())
			.isEqualTo("PP-LCNet_x1_0_textline_ori");
		assertThat(p.getInputWidth()).isEqualTo(c.getInputWidth()).isEqualTo(160);
		assertThat(p.getInputHeight()).isEqualTo(c.getInputHeight()).isEqualTo(80);
		assertThat(p.getChannelOrder()).isEqualTo(c.getChannelOrder())
			.isEqualTo(TextLineChannelOrder.BGR);
		assertThat(p.getMean()).isEqualTo(c.getMean());
		assertThat(p.getStd()).isEqualTo(c.getStd());
		assertThat(p.getInterpolation()).isEqualTo(c.getInterpolation())
			.isEqualTo(TextLineInterpolation.LINEAR);
		assertThat(p.getUpsideDownThreshold()).isEqualTo(c.getUpsideDownThreshold()).isEqualTo(0.5f);
		assertThat(p.isOutputIsProbability()).isEqualTo(c.isOutputIsProbability()).isFalse();
		assertThat(p.getOnnx().getDevice()).as("默认走 CPU").isEqualTo(OrtDevice.CPU);
	}

	@Test
	void kebabCaseKeysShouldBind() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.textline.enabled", "false");
		src.put("mica.ai.textline.model-version", "custom-v1");
		src.put("mica.ai.textline.model-path", "model-tools/textline/models/PP-LCNet_x0_25_textline_ori.onnx");
		src.put("mica.ai.textline.input-width", "192");
		src.put("mica.ai.textline.input-height", "48");
		src.put("mica.ai.textline.channel-order", "RGB");
		src.put("mica.ai.textline.upside-down-threshold", "0.8");
		src.put("mica.ai.textline.output-is-probability", "true");
		src.put("mica.ai.textline.interpolation", "NEAREST");
		src.put("mica.ai.textline.mean[0]", "0.5");
		src.put("mica.ai.textline.mean[1]", "0.5");
		src.put("mica.ai.textline.mean[2]", "0.5");
		src.put("mica.ai.textline.onnx.device", "gpu");
		src.put("mica.ai.textline.onnx.intra-op-num-threads", "4");

		TextLineProperties p = bind(src);

		assertThat(p.isEnabled()).isFalse();
		assertThat(p.getModelVersion()).isEqualTo("custom-v1");
		assertThat(p.getModelPath())
			.isEqualTo("model-tools/textline/models/PP-LCNet_x0_25_textline_ori.onnx");
		assertThat(p.getInputWidth()).isEqualTo(192);
		assertThat(p.getInputHeight()).isEqualTo(48);
		assertThat(p.getChannelOrder()).isEqualTo(TextLineChannelOrder.RGB);
		assertThat(p.getUpsideDownThreshold()).isEqualTo(0.8f);
		assertThat(p.isOutputIsProbability()).isTrue();
		assertThat(p.getInterpolation()).isEqualTo(TextLineInterpolation.NEAREST);
		assertThat(p.getMean()).containsExactly(0.5f, 0.5f, 0.5f);
		assertThat(p.getOnnx().getDevice()).isEqualTo(OrtDevice.GPU);
		assertThat(p.getOnnx().getIntraOpNumThreads()).isEqualTo(4);
	}

	/**
	 * 配置字段最终必须能装进 {@link TextLineConfig} 且通过 {@code validate()}，
	 * 否则 starter 里少写一个 {@code .xxx(properties.getXxx())} 会在启动期炸、或更糟：静默用默认值。
	 */
	@Test
	void propertiesShouldBuildValidCoreConfig() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.textline.model-path", "model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx");
		src.put("mica.ai.textline.channel-order", "RGB");
		src.put("mica.ai.textline.interpolation", "CUBIC");
		src.put("mica.ai.textline.upside-down-threshold", "0.65");
		src.put("mica.ai.textline.output-is-probability", "true");

		TextLineProperties p = bind(src);
		TextLineConfig config = TextLineConfig.builder()
			.modelVersion(p.getModelVersion())
			.modelPath(p.getModelPath())
			.inputWidth(p.getInputWidth())
			.inputHeight(p.getInputHeight())
			.channelOrder(p.getChannelOrder())
			.mean(p.getMean())
			.std(p.getStd())
			.interpolation(p.getInterpolation())
			.upsideDownThreshold(p.getUpsideDownThreshold())
			.outputIsProbability(p.isOutputIsProbability())
			.onnx(p.getOnnx())
			.build();

		config.validate();
		assertThat(config.resolveModelPath())
			.isEqualTo("model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx");
		assertThat(config.getChannelOrder()).isEqualTo(TextLineChannelOrder.RGB);
		assertThat(config.getInterpolation()).isEqualTo(TextLineInterpolation.CUBIC);
		assertThat(config.getUpsideDownThreshold()).isEqualTo(0.65f);
		assertThat(config.isOutputIsProbability()).isTrue();
	}

	/**
	 * 装配一致性：{@link TextLineAutoConfiguration#toConfig} 必须把每个属性都装进
	 * {@link TextLineConfig}。
	 *
	 * <p>调用的就是自动装配真实使用的那段代码（{@code textlineEngine} 内联调它），
	 * <b>不在测试里重抄 builder 链</b>——重抄的写法无法发现「漏写一个
	 * {@code .xxx(properties.getXxx())}」，形同虚设。
	 *
	 * <p>传入非默认值绑定，若字段未被装配，核心配置仍是默认值 ⇒ 断言失败。
	 */
	@Test
	void autoConfigurationBuilderShouldWireEveryProperty() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.textline.model-path", "/tmp/external-textline.onnx");
		src.put("mica.ai.textline.model-version", "PP-LCNet_x0_25_textline_ori");
		src.put("mica.ai.textline.input-width", "160");
		src.put("mica.ai.textline.input-height", "80");
		src.put("mica.ai.textline.channel-order", "RGB");
		src.put("mica.ai.textline.mean[0]", "0.5");
		src.put("mica.ai.textline.mean[1]", "0.5");
		src.put("mica.ai.textline.mean[2]", "0.5");
		src.put("mica.ai.textline.std[0]", "0.25");
		src.put("mica.ai.textline.std[1]", "0.25");
		src.put("mica.ai.textline.std[2]", "0.25");
		src.put("mica.ai.textline.interpolation", "NEAREST");
		src.put("mica.ai.textline.upside-down-threshold", "0.72");
		src.put("mica.ai.textline.output-is-probability", "true");

		TextLineProperties p = bind(src);
		TextLineConfig c = TextLineAutoConfiguration.toConfig(p);

		assertThat(c.getModelVersion()).isEqualTo("PP-LCNet_x0_25_textline_ori");
		assertThat(c.getModelPath()).isEqualTo("/tmp/external-textline.onnx");
		assertThat(c.getInputWidth()).isEqualTo(160);
		assertThat(c.getInputHeight()).isEqualTo(80);
		assertThat(c.getChannelOrder()).as("channel-order 必须被装配")
			.isEqualTo(TextLineChannelOrder.RGB);
		assertThat(c.getMean()).containsExactly(0.5f, 0.5f, 0.5f);
		assertThat(c.getStd()).containsExactly(0.25f, 0.25f, 0.25f);
		assertThat(c.getInterpolation()).isEqualTo(TextLineInterpolation.NEAREST);
		assertThat(c.getUpsideDownThreshold()).as("upside-down-threshold 必须被装配")
			.isEqualTo(0.72f);
		assertThat(c.isOutputIsProbability()).as("output-is-probability 必须被装配").isTrue();
	}

	@Test
	void unsetModelPathShouldFallBackToClasspathTemplate() {
		TextLineConfig config = TextLineConfig.builder().build();
		assertThat(config.resolveModelPath())
			.isEqualTo("classpath:mica-ai/models/textline/PP-LCNet_x1_0_textline_ori.onnx");
	}
}
