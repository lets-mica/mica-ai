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
package net.dreamlu.mica.ai.barcode.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.barcode.core.BarcodeEngine;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * mica-ai-barcode Spring Boot 自动装配（基于 mica-auto）。
 *
 * <p>{@code BarcodeEngine} 不持有 ONNX session / {@code Mat} / 文件句柄，因此 Bean
 * <b>不设 {@code destroyMethod}</b>；它线程安全，整个应用共享一个实例即可。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(BarcodeProperties.class)
@ConditionalOnClass(BarcodeEngine.class)
@ConditionalOnProperty(prefix = "mica.ai.barcode", name = "enabled", havingValue = "true", matchIfMissing = true)
public class BarcodeAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean
	public BarcodeEngine barcodeEngine(BarcodeProperties properties) {
		log.info("mica-ai-barcode 自动装配: formats={} multiScale={} tileScan={}x{} fallback={}",
			properties.getFormats().size(), properties.isMultiScale(),
			properties.getTileRows(), properties.getTileCols(), properties.isGlobalHistogramFallback());
		return BarcodeEngine.create(properties.toConfig());
	}
}
