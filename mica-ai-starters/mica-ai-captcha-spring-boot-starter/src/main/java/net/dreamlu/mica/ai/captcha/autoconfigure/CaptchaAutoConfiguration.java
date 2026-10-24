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

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.captcha.CaptchaEngine;
import net.dreamlu.mica.ai.captcha.config.CaptchaConfig;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * mica-ai-captcha Spring Boot 自动装配（基于 mica-auto）。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(CaptchaProperties.class)
@ConditionalOnClass(CaptchaEngine.class)
@ConditionalOnProperty(prefix = "mica.ai.captcha", name = "enabled", havingValue = "true", matchIfMissing = true)
public class CaptchaAutoConfiguration {

	/**
	 * 创建验证码识别引擎 Bean。
	 *
	 * <p>{@code destroyMethod = "close"} 保证容器关闭时释放底层 ONNX session 与会话选项。
	 *
	 * @param properties 验证码识别配置属性
	 * @return CaptchaEngine 实例
	 */
	@Bean
	@ConditionalOnMissingBean
	public CaptchaEngine captchaEngine(CaptchaProperties properties) {
		return CaptchaEngine.create(toConfig(properties));
	}

	/**
	 * 把 Spring 配置属性翻译成核心模块的 {@link CaptchaConfig}。
	 *
	 * <p>独立成包级静态方法，便于测试直接调用<b>这段真实装配逻辑</b>——
	 * 若在测试里重抄一遍 builder 链，就永远发现不了「漏写一个
	 * {@code .xxx(properties.getXxx())}」这类静默失效。
	 *
	 * @param properties 验证码识别配置属性
	 * @return 核心模块配置对象
	 */
	static CaptchaConfig toConfig(CaptchaProperties properties) {
		return CaptchaConfig.builder()
			.modelPath(properties.getModelPath())
			.charsetPath(properties.getCharsetPath())
			.inputHeight(properties.getInputHeight())
			.normalization(properties.getNormalization())
			.interpolation(properties.getInterpolation())
			.pngFix(properties.isPngFix())
			.onnx(properties.getOnnx())
			.build();
	}
}
