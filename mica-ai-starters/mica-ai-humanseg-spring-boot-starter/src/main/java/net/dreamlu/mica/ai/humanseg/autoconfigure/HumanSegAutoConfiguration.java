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
package net.dreamlu.mica.ai.humanseg.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.humanseg.HumanSegEngine;
import net.dreamlu.mica.ai.humanseg.config.HumanSegConfig;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * mica-ai-humanseg Spring Boot 自动装配（基于 mica-auto）。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(HumanSegProperties.class)
@ConditionalOnClass(HumanSegEngine.class)
@ConditionalOnProperty(prefix = "mica.ai.humanseg", name = "enabled", havingValue = "true", matchIfMissing = true)
public class HumanSegAutoConfiguration {

	/**
	 * 创建人像分割引擎 Bean。
	 *
	 * <p>{@code destroyMethod = "close"} 保证容器关闭时释放底层 ONNX session 与会话选项。
	 *
	 * @param properties 人像分割配置属性
	 * @return HumanSegEngine 实例
	 */
	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public HumanSegEngine humanSegEngine(HumanSegProperties properties) {
		return HumanSegEngine.create(toConfig(properties));
	}

	/**
	 * 把 Spring 配置属性翻译成核心模块的 {@link HumanSegConfig}。
	 *
	 * <p>独立成包级静态方法，便于测试直接验证<b>这段真实装配逻辑</b>，
	 * 避免测试里重抄 builder 链而漏掉映射导致静默失效。
	 *
	 * @param properties 人像分割配置属性
	 * @return 核心模块配置对象
	 */
	static HumanSegConfig toConfig(HumanSegProperties properties) {
		return HumanSegConfig.builder()
			.modelPath(properties.getModelPath())
			.modelVersion(properties.getModelVersion())
			.inputWidth(properties.getInputWidth())
			.inputHeight(properties.getInputHeight())
			.mean(properties.getMean())
			.std(properties.getStd())
			.binaryThreshold(properties.getBinaryThreshold())
			.backgroundColor(properties.getBackgroundColor())
			.onnx(properties.getOnnx())
			.build();
	}
}
