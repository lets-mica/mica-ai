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

class DirectionRuleTest {

	private static Track track() {
		return new Track(1, TrackFixture.box(0f, 50f), 5, 0, 64, 0, 0L);
	}

	@Test
	void uTurnEmitsOneEvent() {
		DirectionRule rule = new DirectionRule("u-turn");
		Track track = track();

		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L))).isEmpty();
		TrackFixture.move(track, 1, 100L, 50f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L))).isEmpty();

		TrackFixture.move(track, 2, 200L, 10f, 50f);
		List<TrackEvent> events = TrackFixture.run(rule, TrackFixture.frame(track, 2, 200L));
		assertThat(events).hasSize(1);
		assertThat(events.get(0).getType()).isEqualTo(TrackEventType.DIRECTION_CHANGED);
		assertThat(events.get(0).getDirection()).isEqualTo("左");
		assertThat(events.get(0).getDetail()).contains("turnAngle=180.0", "from=右", "to=左");
	}

	@Test
	void straightMovementDoesNotEmit() {
		DirectionRule rule = new DirectionRule("u-turn");
		Track track = track();

		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.move(track, 1, 100L, 50f, 50f);
		TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L));
		TrackFixture.move(track, 2, 200L, 100f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 2, 200L))).isEmpty();
	}

	@Test
	void cooldownSuppressesRepeatedReports() {
		DirectionRule rule = new DirectionRule("u-turn", 120d, 5f, 10, true);
		Track track = track();

		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.move(track, 1, 100L, 50f, 50f);
		TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L));
		TrackFixture.move(track, 2, 200L, 10f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 2, 200L))).hasSize(1);

		TrackFixture.move(track, 3, 300L, 60f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 3, 300L))).isEmpty();
	}

	@Test
	void tinyJitterIsFilteredByMinDisplacement() {
		DirectionRule rule = new DirectionRule("u-turn", 120d, 20f, 0, true);
		Track track = track();

		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.move(track, 1, 100L, 10f, 50f);
		TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L));
		TrackFixture.move(track, 2, 200L, 0f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 2, 200L))).isEmpty();
	}

	@Test
	void resetAndValidation() {
		DirectionRule rule = new DirectionRule("u-turn", 120d, 5f, 0, true);
		Track track = track();
		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		TrackFixture.move(track, 1, 100L, 50f, 50f);
		TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L));
		TrackFixture.move(track, 2, 200L, 10f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 2, 200L))).hasSize(1);

		rule.reset();
		TrackFixture.move(track, 3, 300L, 60f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 3, 300L))).hasSize(1);

		assertThatThrownBy(() -> new DirectionRule("x", 200d, 5f, 1, true))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("turnAngleDegrees");
	}
}
