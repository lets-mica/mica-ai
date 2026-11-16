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

import net.dreamlu.mica.ai.track.model.Track;
import net.dreamlu.mica.ai.track.model.TrackEvent;
import net.dreamlu.mica.ai.track.model.TrackEventType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class DwellRuleTest {

	private static final Roi ZONE = Roi.rectangle(0f, 0f, 100f, 100f);

	private static DwellRule rule() {
		return new DwellRule("zone", ZONE, 1000L);
	}

	@Test
	void dwellUsesRealMillisecondsNotFrameCount() {
		DwellRule rule = rule();
		Track track = TrackFixture.track(1, 50f, 50f, 5);

		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L))).isEmpty();
		TrackFixture.move(track, 1, 3000L, 50f, 50f);
		List<TrackEvent> events = TrackFixture.run(rule, TrackFixture.frame(track, 1, 3000L));
		assertThat(events).hasSize(1);
		assertThat(events.get(0).getType()).isEqualTo(TrackEventType.DWELL_EXCEEDED);
		assertThat(events.get(0).getDetail()).contains("stayMs=3000");
	}

	@Test
	void reportsOnceThenRearmsAfterLeaving() {
		DwellRule rule = rule();
		Track track = TrackFixture.track(1, 50f, 50f, 5);

		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.move(track, 1, 500L, 50f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 1, 500L))).isEmpty();

		TrackFixture.move(track, 2, 1200L, 50f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 2, 1200L))).hasSize(1);
		TrackFixture.move(track, 3, 1500L, 50f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 3, 1500L))).isEmpty();

		TrackFixture.move(track, 4, 1600L, 150f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 4, 1600L))).isEmpty();
		TrackFixture.move(track, 5, 1700L, 50f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 5, 1700L))).isEmpty();
		TrackFixture.move(track, 6, 2800L, 50f, 50f);
		List<TrackEvent> again = TrackFixture.run(rule, TrackFixture.frame(track, 6, 2800L));
		assertThat(again).hasSize(1);
		assertThat(again.get(0).getDetail()).contains("stayMs=1100");
	}

	@Test
	void missingTimestampFailsFastInsteadOfGuessingDuration() {
		DwellRule rule = rule();
		Track track = TrackFixture.track(1, 50f, 50f, 5);
		assertThatThrownBy(() -> TrackFixture.run(rule, TrackFixture.frame(track, 0, -1L)))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("真实时间戳");
	}

	@Test
	void resetClearsPendingTimers() {
		DwellRule rule = rule();
		Track track = TrackFixture.track(1, 50f, 50f, 5);
		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		rule.reset();

		TrackFixture.move(track, 1, 3000L, 50f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 1, 3000L))).isEmpty();
	}

	@Test
	void invalidThresholdFailsFast() {
		assertThatThrownBy(() -> new DwellRule("x", ZONE, 0L))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("thresholdMs");
	}
}
