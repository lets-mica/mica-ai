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
package net.dreamlu.mica.ai.track.model;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * 轨迹上的单个采样点（不可变）。
 *
 * <p>{@code observed=true} 表示该帧有实测检测框命中；{@code false} 表示只有 Kalman 预测框。
 * 行为规则默认只用实测点，避免遮挡期的预测框造成误报。
 */
@Getter
@ToString
@RequiredArgsConstructor
public class TrackPoint {

	private final int frameIndex;
	private final long timestampMs;
	private final float cx;
	private final float cy;
	private final float x1;
	private final float y1;
	private final float x2;
	private final float y2;
	private final boolean observed;

	public boolean hasTimestamp() {
		return timestampMs >= 0L;
	}
}
