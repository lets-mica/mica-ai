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

import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;
import net.dreamlu.mica.ai.track.model.TrackEvent;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

final class TrackFixture {

	static final int CONFIRM_AGE = 2;

	private TrackFixture() {
	}

	static BBox box(float cx, float cy) {
		return new BBox(cx - 5f, cy - 5f, cx + 5f, cy + 5f, 0.9f, 3);
	}

	static Track track(int id, float cx, float cy, int age) {
		return new Track(id, box(cx, cy), age, 0, 64, 0, 0L);
	}

	static void move(Track track, int frameIndex, long timestampMs, float cx, float cy) {
		track.appendPoint(frameIndex, timestampMs, box(cx, cy), true);
	}

	static void predict(Track track, int frameIndex, long timestampMs, float cx, float cy) {
		track.appendPoint(frameIndex, timestampMs, box(cx, cy), false);
	}

	static FrameContext frame(Track track, int frameIndex, long timestampMs) {
		return new FrameContext(frameIndex, timestampMs, Collections.singletonList(track), CONFIRM_AGE);
	}

	static FrameContext emptyFrame(int frameIndex, long timestampMs) {
		return new FrameContext(frameIndex, timestampMs,
			Collections.<Track>emptyList(), CONFIRM_AGE);
	}

	static List<TrackEvent> run(TrackEventRule rule, FrameContext context) {
		List<TrackEvent> events = new ArrayList<>();
		rule.onFrame(context, events);
		return events;
	}
}
