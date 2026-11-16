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

import net.dreamlu.mica.ai.track.config.ByteTrackConfig;
import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;
import net.dreamlu.mica.ai.track.model.TrackEvent;
import net.dreamlu.mica.ai.track.model.TrackEventType;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TrackEventEngineTest {

	private static ByteTrackConfig config() {
		return ByteTrackConfig.builder()
			.highThreshold(0.5f)
			.lowThreshold(0.01f)
			.matchThresh(0.1f)
			.matchLowThresh(0.1f)
			.confirmAge(1)
			.trackBufferedThresh(10)
			.historySize(32)
			.build();
	}

	private static BBox box(float cx) {
		return new BBox(cx - 30f, 20f, cx + 30f, 80f, 0.9f, 7);
	}

	private static List<BBox> one(float cx) {
		return Collections.singletonList(box(cx));
	}

	private static TrackEventEngine engine(LineCrossingRule line, DwellRule dwell) {
		return TrackEventEngine.create(config(), line, dwell);
	}

	private static LineCrossingRule line() {
		return new LineCrossingRule("gate", 100f, 0f, 100f, 400f,
			LineCrossingRule.Direction.BOTH, false, true);
	}

	private static DwellRule dwell() {
		return new DwellRule("zone", Roi.rectangle(0f, 0f, 200f, 200f),
			1000L, AnchorPolicy.CENTER, false);
	}

	@Test
	void endToEndProducesStableTracksVelocityAndEvents() {
		TrackEventEngine engine = engine(line(), dwell());

		FrameResult r0 = engine.process(one(20f), 0L);
		assertThat(r0.getFrameIndex()).isEqualTo(1);
		assertThat(r0.getTracks()).hasSize(1);
		assertThat(r0.getEvents()).isEmpty();

		engine.process(one(40f), 300L);
		engine.process(one(60f), 600L);
		engine.process(one(80f), 900L);
		FrameResult r4 = engine.process(one(100f), 1200L);

		assertThat(r4.getEvents()).hasSize(1);
		assertThat(r4.getEvents().get(0).getType()).isEqualTo(TrackEventType.DWELL_EXCEEDED);
		Track track = r4.getTracks().get(0);
		assertThat(track.getId()).isEqualTo(1);
		assertThat(track.getClassId()).isEqualTo(7);
		assertThat(track.getFirstSeenMs()).isZero();
		assertThat(track.getLastSeenMs()).isEqualTo(1200L);
		assertThat(track.getPoints()).hasSize(5);
		assertThat(track.velocityXPerSecond()).isGreaterThan(0d);
		assertThat(track.directionLabel()).isEqualTo("右");

		FrameResult r5 = engine.process(one(120f), 1500L);
		List<TrackEvent> crossing = r5.getEvents();
		assertThat(crossing).hasSize(1);
		assertThat(crossing.get(0).getType()).isEqualTo(TrackEventType.LINE_CROSSED);
		assertThat(crossing.get(0).getDirection()).isEqualTo(LineCrossingRule.DIR_BACKWARD);
	}

	@Test
	void timestampMustBeMonotonicallyNonDecreasing() {
		TrackEventEngine engine = engine(line(), dwell());
		engine.process(one(20f), 1000L);
		assertThatThrownBy(() -> engine.process(one(40f), 500L))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("单调不减");
	}

	@Test
	void legacyApiWithDwellRuleFailsFast() {
		TrackEventEngine engine = engine(line(), dwell());
		assertThatThrownBy(() -> engine.process(one(20f)))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("真实时间戳");
	}

	@Test
	void noRulesStillTracksFrames() {
		TrackEventEngine engine = TrackEventEngine.create(config());
		FrameResult result = engine.process(one(20f), 0L);
		assertThat(result.getEvents()).isEmpty();
		assertThat(result.getTracks()).hasSize(1);
		assertThat(result.getTimestampMs()).isZero();
		assertThat(engine.getRules()).isEmpty();
	}

	@Test
	void resetClearsTrackerAndRuleState() {
		LineCrossingRule line = line();
		TrackEventEngine engine = engine(line, dwell());
		engine.process(one(20f), 0L);
		assertThat(line.watchedTracks()).isEqualTo(1);

		engine.reset();
		assertThat(line.watchedTracks()).isZero();
		FrameResult result = engine.process(one(20f), 5000L);
		assertThat(result.getFrameIndex()).isEqualTo(1);
		assertThat(result.getTracks().get(0).getId()).isEqualTo(1);
		assertThat(result.getTracks().get(0).getFirstSeenMs()).isEqualTo(5000L);
	}

	@Test
	void invalidArgumentsFailFast() {
		assertThatThrownBy(() -> TrackEventEngine.create(config(), (TrackEventRule) null))
			.isInstanceOf(IllegalArgumentException.class);
		TrackEventEngine engine = TrackEventEngine.create(config());
		assertThatThrownBy(() -> engine.process(one(20f), -2L))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("timestampMs");
	}
}
