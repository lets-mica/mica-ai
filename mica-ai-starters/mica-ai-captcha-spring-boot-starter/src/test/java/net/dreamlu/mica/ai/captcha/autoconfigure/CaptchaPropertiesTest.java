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
package net.dreamlu.mica.ai.captcha.autoconfigure;

import net.dreamlu.mica.ai.captcha.config.CaptchaConfig;
import net.dreamlu.mica.ai.captcha.config.CaptchaInterpolation;
import net.dreamlu.mica.ai.captcha.config.CaptchaNormalization;
import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link CaptchaProperties} 配置绑定测试（纯绑定，不启动 Spring 上下文、不加载模型）。
 *
 * <p>验证目标：文档里写的 {@code mica.ai.captcha.*} 键名真的能绑上、
 * 默认值与 {@link CaptchaConfig} 保持一致、嵌套 {@code onnx} 能绑到 {@link OrtDevice}。
 * 这些是最容易「文档与代码不一致」的地方（改字段名忘了改 yml 即静默失效）。
 */
class CaptchaPropertiesTest {

	private static CaptchaProperties bind(Map<String, Object> source) {
		Binder binder = new Binder(new MapConfigurationPropertySource(source));
		return binder.bind("mica.ai.captcha", Bindable.of(CaptchaProperties.class))
			.orElseGet(CaptchaProperties::new);
	}

	@Test
	void defaultsShouldMatchCoreConfigDefaults() {
		CaptchaProperties p = new CaptchaProperties();
		CaptchaConfig c = CaptchaConfig.builder()
			.modelPath("model.onnx")
			.charsetPath("charset.txt")
			.build();

		assertThat(p.isEnabled()).as("默认应启用").isTrue();
		assertThat(p.getInputHeight()).isEqualTo(c.getInputHeight()).isEqualTo(64);
		assertThat(p.getNormalization()).isEqualTo(c.getNormalization())
			.isEqualTo(CaptchaNormalization.DIV_255);
		assertThat(p.getInterpolation()).isEqualTo(c.getInterpolation())
			.isEqualTo(CaptchaInterpolation.BILINEAR);
		assertThat(p.isPngFix()).isEqualTo(c.isPngFix()).isTrue();
		assertThat(p.getOnnx().getDevice()).as("默认走 CPU").isEqualTo(OrtDevice.CPU);
	}

	@Test
	void kebabCaseKeysShouldBind() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.captcha.enabled", "false");
		src.put("mica.ai.captcha.model-path", "/data/models/ddddocr.onnx");
		src.put("mica.ai.captcha.charset-path", "/data/models/charset.txt");
		src.put("mica.ai.captcha.input-height", "48");
		src.put("mica.ai.captcha.normalization", "MEAN_0_5");
		src.put("mica.ai.captcha.interpolation", "BICUBIC");
		src.put("mica.ai.captcha.png-fix", "false");
		src.put("mica.ai.captcha.onnx.device", "gpu");
		src.put("mica.ai.captcha.onnx.intra-op-num-threads", "4");

		CaptchaProperties p = bind(src);

		assertThat(p.isEnabled()).isFalse();
		assertThat(p.getModelPath()).isEqualTo("/data/models/ddddocr.onnx");
		assertThat(p.getCharsetPath()).isEqualTo("/data/models/charset.txt");
		assertThat(p.getInputHeight()).isEqualTo(48);
		assertThat(p.getNormalization()).isEqualTo(CaptchaNormalization.MEAN_0_5);
		assertThat(p.getInterpolation()).isEqualTo(CaptchaInterpolation.BICUBIC);
		assertThat(p.isPngFix()).isFalse();
		assertThat(p.getOnnx().getDevice()).isEqualTo(OrtDevice.GPU);
		assertThat(p.getOnnx().getIntraOpNumThreads()).isEqualTo(4);
	}

	/**
	 * 装配一致性：{@link CaptchaAutoConfiguration#toConfig} 必须把每个属性都装进
	 * {@link CaptchaConfig}。
	 *
	 * <p>调用的就是自动装配真实使用的那段代码，<b>不在测试里重抄 builder 链</b>——
	 * 重抄的写法无法发现「漏写一个 {@code .xxx(properties.getXxx())}」，形同虚设。
	 *
	 * <p>传入非默认值绑定，若字段未被装配，核心配置仍是默认值 ⇒ 断言失败。
	 */
	@Test
	void autoConfigurationBuilderShouldWireEveryProperty() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.captcha.model-path", "/tmp/external-captcha.onnx");
		src.put("mica.ai.captcha.charset-path", "/tmp/external-charset.txt");
		src.put("mica.ai.captcha.input-height", "48");
		src.put("mica.ai.captcha.normalization", "MEAN_0_5");
		src.put("mica.ai.captcha.interpolation", "NEAREST");
		src.put("mica.ai.captcha.png-fix", "false");

		CaptchaProperties p = bind(src);
		CaptchaConfig c = CaptchaAutoConfiguration.toConfig(p);

		assertThat(c.getModelPath()).isEqualTo("/tmp/external-captcha.onnx");
		assertThat(c.getCharsetPath()).as("charset-path 必须被装配")
			.isEqualTo("/tmp/external-charset.txt");
		assertThat(c.getInputHeight()).as("input-height 必须被装配").isEqualTo(48);
		assertThat(c.getNormalization()).as("normalization 必须被装配")
			.isEqualTo(CaptchaNormalization.MEAN_0_5);
		assertThat(c.getInterpolation()).as("interpolation 必须被装配")
			.isEqualTo(CaptchaInterpolation.NEAREST);
		assertThat(c.isPngFix()).as("png-fix 必须被装配").isFalse();
	}
}
