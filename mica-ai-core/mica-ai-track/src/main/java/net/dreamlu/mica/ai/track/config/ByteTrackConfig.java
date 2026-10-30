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
package net.dreamlu.mica.ai.track.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

/**
 * ByteTrack 配置（默认值与 ifzhang/ByteTrack 原版对齐）。
 *
 * <p>IoU 阈值用于 HungarianMatcher 的成本矩阵：阈值外设 {@code INF}，
 * 低于阈值才参与匹配。
 *
 * <p>所有配置在构造时冻结，不可变。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class ByteTrackConfig {

	/**
	 * 高分框 / 低分框分界线。高于此 score 的 det 走第一轮匹配；
	 * 低于但高于 lowThreshold 的走第二轮（ByteTrack 灵魂）。
	 */
	@Builder.Default
	private float highThreshold = 0.6f;

	/**
	 * 低分框丢弃阈值（低于该值直接丢弃，不参与任何匹配）。
	 */
	@Builder.Default
	private float lowThreshold = 0.1f;

	/**
	 * 第一轮（高分 det ↔ 成熟轨迹）IoU 匹配阈值。
	 */
	@Builder.Default
	private float matchThresh = 0.8f;

	/**
	 * 第二轮（低分 det ↔ 未匹配轨迹）IoU 匹配阈值。
	 */
	@Builder.Default
	private float matchLowThresh = 0.5f;

	/**
	 * 连续多少帧未匹配的轨迹丢弃（帧缓存长度）。
	 */
	@Builder.Default
	private int trackBufferedThresh = 30;

	/**
	 * 新轨迹需要累计多少帧匹配才算"成熟"（才会被第一轮匹配考虑）。
	 */
	@Builder.Default
	private int confirmAge = 3;

	public void validate() {
		if (highThreshold <= lowThreshold) {
			throw new IllegalArgumentException("highThreshold 必须 > lowThreshold");
		}
		if (matchThresh <= 0f || matchThresh > 1f) {
			throw new IllegalArgumentException("matchThresh 必须在 (0, 1]");
		}
		if (matchLowThresh <= 0f || matchLowThresh > 1f) {
			throw new IllegalArgumentException("matchLowThresh 必须在 (0, 1]");
		}
		if (trackBufferedThresh <= 0) {
			throw new IllegalArgumentException("trackBufferedThresh 必须为正整数");
		}
		if (confirmAge <= 0) {
			throw new IllegalArgumentException("confirmAge 必须为正整数");
		}
	}
}
