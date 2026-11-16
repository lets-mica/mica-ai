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
 * 单帧产生的轨迹事件（不可变）。
 *
 * <p>{@code direction} 仅在线穿越 / 折返事件里有值，其余为空串；
 * {@code detail} 是人类可读的补充信息（驻留毫秒数、转角度数、累计计数等）。
 */
@Getter
@ToString
@RequiredArgsConstructor
public class TrackEvent {

	private final TrackEventType type;
	private final int trackId;
	private final int classId;
	private final int frameIndex;
	private final long timestampMs;
	private final String ruleName;
	private final String direction;
	private final String detail;
	private final BBox box;
}
