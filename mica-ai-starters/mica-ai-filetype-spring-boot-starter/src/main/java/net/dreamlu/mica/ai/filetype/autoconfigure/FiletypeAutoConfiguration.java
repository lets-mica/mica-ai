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

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.filetype.config.FiletypeConfig;
import net.dreamlu.mica.ai.filetype.detection.FiletypeDetector;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * mica-ai-filetype Spring Boot 自动装配（基于 mica-auto）。
 *
 * <p>由 {@code mica-auto} 扫描本类上的 {@code @Component} 注解，自动生成
 * {@code META-INF/spring.factories} 中的 {@code EnableAutoConfiguration} 条目。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(FiletypeProperties.class)
@ConditionalOnClass(FiletypeDetector.class)
@ConditionalOnProperty(prefix = "mica.ai.filetype", name = "enabled", havingValue = "true", matchIfMissing = true)
public class FiletypeAutoConfiguration {

	/**
	 * 创建文件类型检测器 Bean。
	 *
	 * @param properties 文件类型配置属性
	 * @return FiletypeDetector 实例
	 */
	@Bean
	@ConditionalOnMissingBean
	public FiletypeDetector filetypeDetector(FiletypeProperties properties) {
		return FiletypeDetector.create(toConfig(properties));
	}

	/**
	 * 把 Spring 配置属性翻译成核心模块的 {@link FiletypeConfig}。
	 *
	 * <p>独立成包级静态方法，便于测试直接调用<b>这段真实装配逻辑</b>——
	 * 若在测试里重抄一遍 builder 链，就永远发现不了「漏写一个
	 * {@code .xxx(properties.getXxx())}」这类静默失效。
	 *
	 * @param properties 文件类型配置属性
	 * @return 核心模块配置对象
	 */
	static FiletypeConfig toConfig(FiletypeProperties properties) {
		return FiletypeConfig.builder()
			.modelVersion(properties.getModelVersion())
			.modelPath(properties.getModelPath())
			.configPath(properties.getConfigPath())
			.contentTypesPath(properties.getContentTypesPath())
			.predictionMode(properties.getPredictionMode())
			.onnx(properties.getOnnx())
			.build();
	}

}
