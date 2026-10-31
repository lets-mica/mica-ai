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

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.track.config.ByteTrackConfig;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * mica-ai-track 配置属性，对应 {@code mica.ai.track} 前缀。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.track")
public class TrackProperties {

	/**
	 * 是否启用多目标跟踪（false 时不装配 ByteTracker Bean）。
	 */
	private boolean enabled = true;

	/**
	 * 高分框 / 低分框分界线。
	 */
	private float highThreshold = 0.6f;

	/**
	 * 低分框丢弃阈值。
	 */
	private float lowThreshold = 0.1f;

	/**
	 * 第一轮（高分 det ↔ 成熟轨迹）IoU 匹配阈值。
	 */
	private float matchThresh = 0.8f;

	/**
	 * 第二轮（低分 det ↔ 未匹配轨迹）IoU 匹配阈值。
	 */
	private float matchLowThresh = 0.5f;

	/**
	 * 连续多少帧未匹配的轨迹丢弃。
	 */
	private int trackBufferedThresh = 30;

	/**
	 * 新轨迹成熟所需帧数。
	 */
	private int confirmAge = 3;

	public ByteTrackConfig toConfig() {
		return ByteTrackConfig.builder()
			.highThreshold(highThreshold)
			.lowThreshold(lowThreshold)
			.matchThresh(matchThresh)
			.matchLowThresh(matchLowThresh)
			.trackBufferedThresh(trackBufferedThresh)
			.confirmAge(confirmAge)
			.build();
	}
}
