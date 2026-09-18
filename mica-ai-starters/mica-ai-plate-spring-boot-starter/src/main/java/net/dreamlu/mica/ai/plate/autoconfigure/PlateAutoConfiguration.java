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
package net.dreamlu.mica.ai.plate.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.plate.config.PlateConfig;
import net.dreamlu.mica.ai.plate.pipeline.PlatePipeline;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * mica-ai-plate Spring Boot 自动装配（基于 mica-auto）。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(PlateProperties.class)
@ConditionalOnClass(PlatePipeline.class)
@ConditionalOnProperty(prefix = "mica.ai.plate", name = "enabled", havingValue = "true", matchIfMissing = true)
public class PlateAutoConfiguration {

	/**
	 * 创建车牌识别管道 Bean。
	 *
	 * @param properties 车牌识别配置属性
	 * @return PlatePipeline 实例
	 */
	@Bean
	@ConditionalOnMissingBean
	public PlatePipeline platePipeline(PlateProperties properties) {
		PlateConfig config = PlateConfig.builder()
			.modelVersion(properties.getModelVersion())
			.detectionModelPath(properties.getDetectionModelPath())
			.recognitionModelPath(properties.getRecognitionModelPath())
			.classificationModelPath(properties.getClassificationModelPath())
			.detectionInputSize(properties.getDetectionInputSize())
			.recognitionInputHeight(properties.getRecognitionInputHeight())
			.recognitionInputWidth(properties.getRecognitionInputWidth())
			.classificationInputSize(properties.getClassificationInputSize())
			.detectionConfidenceThreshold(properties.getDetectionConfidenceThreshold())
			.detectionNmsThreshold(properties.getDetectionNmsThreshold())
			.maxPlates(properties.getMaxPlates())
			.onnx(properties.getOnnx())
			.build();
		return PlatePipeline.create(config);
	}
}
