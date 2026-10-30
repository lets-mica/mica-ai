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
package net.dreamlu.mica.ai.enhance.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.enhance.config.EnhanceConfig;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * mica-ai-enhance Spring Boot 自动装配（基于 mica-auto）。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(EnhanceProperties.class)
@ConditionalOnClass(EnhanceEngine.class)
@ConditionalOnProperty(prefix = "mica.ai.enhance", name = "enabled", havingValue = "true", matchIfMissing = true)
public class EnhanceAutoConfiguration {

	/**
	 * 创建图像超分引擎 Bean。
	 *
	 * <p>{@code destroyMethod = "close"} 保证容器关闭时释放底层 ONNX session 与会话选项。
	 *
	 * @param properties 超分配置属性
	 * @return EnhanceEngine 实例
	 */
	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public EnhanceEngine enhanceEngine(EnhanceProperties properties) {
		return EnhanceEngine.create(toConfig(properties));
	}

	/**
	 * 把 Spring 配置属性翻译成核心模块的 {@link EnhanceConfig}。
	 *
	 * <p>独立成包级静态方法，便于测试直接验证<b>这段真实装配逻辑</b>，
	 * 避免测试里重抄 builder 链而漏掉映射导致静默失效。
	 *
	 * @param properties 超分配置属性
	 * @return 核心模块配置对象
	 */
	static EnhanceConfig toConfig(EnhanceProperties properties) {
		return EnhanceConfig.builder()
			.modelPath(properties.getModelPath())
			.modelVersion(properties.getModelVersion())
			.scale(properties.getScale())
			.tileSize(properties.getTileSize())
			.tileOverlap(properties.getTileOverlap())
			.onnx(properties.getOnnx())
			.build();
	}
}
