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

class LineCrossingRuleTest {

	private static LineCrossingRule verticalLine() {
		return new LineCrossingRule("gate", 100f, 0f, 100f, 200f);
	}

	@Test
	void crossingEmitsOneEventWithDirection() {
		LineCrossingRule rule = verticalLine();
		Track track = TrackFixture.track(1, 50f, 50f, 5);

		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L))).isEmpty();
		TrackFixture.move(track, 1, 100L, 150f, 50f);
		List<TrackEvent> events = TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L));

		assertThat(events).hasSize(1);
		TrackEvent event = events.get(0);
		assertThat(event.getType()).isEqualTo(TrackEventType.LINE_CROSSED);
		assertThat(event.getDirection()).isEqualTo(LineCrossingRule.DIR_BACKWARD);
		assertThat(event.getRuleName()).isEqualTo("gate");
		assertThat(event.getTrackId()).isEqualTo(1);
		assertThat(event.getClassId()).isEqualTo(3);
		assertThat(event.getTimestampMs()).isEqualTo(100L);
		assertThat(event.getBox().getX1()).isEqualTo(145f);
	}

	@Test
	void directionFilterOnlyCountsConfiguredSide() {
		LineCrossingRule rule = new LineCrossingRule("gate", 100f, 0f, 100f, 200f,
			LineCrossingRule.Direction.FORWARD, true, true);
		Track track = TrackFixture.track(1, 50f, 50f, 5);

		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.move(track, 1, 100L, 150f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L))).isEmpty();

		TrackFixture.move(track, 2, 200L, 50f, 50f);
		List<TrackEvent> events = TrackFixture.run(rule, TrackFixture.frame(track, 2, 200L));
		assertThat(events).hasSize(1);
		assertThat(events.get(0).getDirection()).isEqualTo(LineCrossingRule.DIR_FORWARD);
	}

	@Test
	void stayingOnOneSideDoesNotEmit() {
		LineCrossingRule rule = verticalLine();
		Track track = TrackFixture.track(1, 50f, 50f, 5);

		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.move(track, 1, 100L, 60f, 50f);
		TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L));
		TrackFixture.move(track, 2, 200L, 70f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 2, 200L))).isEmpty();
	}

	@Test
	void immatureTrackIsIgnored() {
		LineCrossingRule rule = verticalLine();
		Track track = TrackFixture.track(1, 50f, 50f, 1);

		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.move(track, 1, 100L, 150f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L))).isEmpty();
	}

	@Test
	void predictedPointDoesNotTriggerCrossing() {
		LineCrossingRule rule = verticalLine();
		Track track = TrackFixture.track(1, 50f, 50f, 5);

		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.predict(track, 1, 100L, 150f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L))).isEmpty();
	}

	@Test
	void backAndForthProducesTwoEventsAndResetClearsState() {
		LineCrossingRule rule = verticalLine();
		Track track = TrackFixture.track(1, 50f, 50f, 5);

		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.move(track, 1, 100L, 150f, 50f);
		List<TrackEvent> first = TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L));
		TrackFixture.move(track, 2, 200L, 50f, 50f);
		List<TrackEvent> second = TrackFixture.run(rule, TrackFixture.frame(track, 2, 200L));
		assertThat(first).hasSize(1);
		assertThat(first.get(0).getDirection()).isEqualTo(LineCrossingRule.DIR_BACKWARD);
		assertThat(second).hasSize(1);
		assertThat(second.get(0).getDirection()).isEqualTo(LineCrossingRule.DIR_FORWARD);
		assertThat(rule.watchedTracks()).isEqualTo(1);

		rule.reset();
		assertThat(rule.watchedTracks()).isZero();
	}

	@Test
	void degenerateLineFailsFast() {
		assertThatThrownBy(() -> new LineCrossingRule("bad", 10f, 10f, 10f, 10f))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("端点");
	}
}
