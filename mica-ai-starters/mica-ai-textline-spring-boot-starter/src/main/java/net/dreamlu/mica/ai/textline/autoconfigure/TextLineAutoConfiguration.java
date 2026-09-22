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

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.textline.TextLineEngine;
import net.dreamlu.mica.ai.textline.config.TextLineConfig;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * mica-ai-textline Spring Boot 自动装配（基于 mica-auto）。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(TextLineProperties.class)
@ConditionalOnClass(TextLineEngine.class)
@ConditionalOnProperty(prefix = "mica.ai.textline", name = "enabled", havingValue = "true", matchIfMissing = true)
public class TextLineAutoConfiguration {

	/**
	 * 创建文本行方向分类引擎 Bean。
	 *
	 * <p>{@code destroyMethod = "close"} 保证容器关闭时释放底层 ONNX session 与会话选项。
	 *
	 * @param properties 方向分类配置属性
	 * @return TextLineEngine 实例
	 */
	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public TextLineEngine textlineEngine(TextLineProperties properties) {
		return TextLineEngine.create(toConfig(properties));
	}

	/**
	 * 把 Spring 配置属性翻译成核心模块的 {@link TextLineConfig}。
	 *
	 * <p>独立成包级静态方法，便于测试直接调用<b>这段真实装配逻辑</b>——
	 * 若在测试里重抄一遍 builder 链，就永远发现不了「漏写一个
	 * {@code .xxx(properties.getXxx())}」这类静默失效。
	 *
	 * @param properties 方向分类配置属性
	 * @return 核心模块配置对象
	 */
	static TextLineConfig toConfig(TextLineProperties properties) {
		return TextLineConfig.builder()
			.modelVersion(properties.getModelVersion())
			.modelPath(properties.getModelPath())
			.inputWidth(properties.getInputWidth())
			.inputHeight(properties.getInputHeight())
			.channelOrder(properties.getChannelOrder())
			.mean(properties.getMean())
			.std(properties.getStd())
			.interpolation(properties.getInterpolation())
			.upsideDownThreshold(properties.getUpsideDownThreshold())
			.outputIsProbability(properties.isOutputIsProbability())
			.onnx(properties.getOnnx())
			.build();
	}
}
