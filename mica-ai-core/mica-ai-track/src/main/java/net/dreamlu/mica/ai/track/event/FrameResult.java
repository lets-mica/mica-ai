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
package net.dreamlu.mica.ai.track.event;

import lombok.Getter;
import net.dreamlu.mica.ai.track.model.Track;
import net.dreamlu.mica.ai.track.model.TrackEvent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 一帧的完整输出：活跃轨迹 + 该帧触发的事件（不可变快照）。
 */
@Getter
public class FrameResult {

	private final int frameIndex;
	private final long timestampMs;
	private final List<Track> tracks;
	private final List<TrackEvent> events;

	public FrameResult(int frameIndex, long timestampMs, List<Track> tracks, List<TrackEvent> events) {
		this.frameIndex = frameIndex;
		this.timestampMs = timestampMs;
		this.tracks = Collections.unmodifiableList(new ArrayList<>(tracks));
		this.events = Collections.unmodifiableList(new ArrayList<>(events));
	}
}
