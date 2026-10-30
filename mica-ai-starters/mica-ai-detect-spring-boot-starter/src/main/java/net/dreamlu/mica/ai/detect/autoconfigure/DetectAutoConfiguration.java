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
package net.dreamlu.mica.ai.detect.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.detect.DetectEngine;
import net.dreamlu.mica.ai.detect.config.DetectConfig;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * mica-ai-detect Spring Boot 自动装配（基于 mica-auto）。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(DetectProperties.class)
@ConditionalOnClass(DetectEngine.class)
@ConditionalOnProperty(prefix = "mica.ai.detect", name = "enabled", havingValue = "true", matchIfMissing = true)
public class DetectAutoConfiguration {

	/**
	 * 创建目标检测引擎 Bean。
	 *
	 * <p>{@code destroyMethod = "close"} 保证容器关闭时释放底层 ONNX session 与会话选项。
	 *
	 * @param properties 检测配置属性
	 * @return DetectEngine 实例
	 */
	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public DetectEngine detectEngine(DetectProperties properties) {
		return DetectEngine.create(toConfig(properties));
	}

	/**
	 * 把 Spring 配置属性翻译成核心模块的 {@link DetectConfig}。
	 *
	 * <p>独立成包级静态方法，便于测试直接调用<b>这段真实装配逻辑</b>，
	 * 避免测试里重抄 builder 链而漏掉映射导致静默失效。
	 *
	 * @param properties 检测配置属性
	 * @return 核心模块配置对象
	 */
	static DetectConfig toConfig(DetectProperties properties) {
		return DetectConfig.builder()
			.modelPath(properties.getModelPath())
			.family(properties.getFamily())
			.inputSize(properties.getInputSize())
			.scoreThreshold(properties.getScoreThreshold())
			.nmsThreshold(properties.getNmsThreshold())
			.maxDetections(properties.getMaxDetections())
			.labels(properties.getLabels())
			.onnx(properties.getOnnx())
			.build();
	}
}
