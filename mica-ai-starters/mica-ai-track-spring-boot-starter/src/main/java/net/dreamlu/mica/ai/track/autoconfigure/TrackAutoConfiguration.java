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
package net.dreamlu.mica.ai.track.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.track.core.ByteTracker;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * mica-ai-track Spring Boot 自动装配。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(TrackProperties.class)
@ConditionalOnClass(ByteTracker.class)
@ConditionalOnProperty(prefix = "mica.ai.track", name = "enabled", havingValue = "true", matchIfMissing = true)
public class TrackAutoConfiguration {

	@Bean
	@ConditionalOnMissingBean
	public ByteTracker byteTracker(TrackProperties properties) {
		log.info("mica-ai-track 初始化完成: high={}, low={}, match={}, matchLow={}, confirmAge={}, buffered={}",
			properties.getHighThreshold(), properties.getLowThreshold(),
			properties.getMatchThresh(), properties.getMatchLowThresh(),
			properties.getConfirmAge(), properties.getTrackBufferedThresh());
		return new ByteTracker(properties.toConfig());
	}
}
