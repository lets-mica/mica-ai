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
import net.dreamlu.mica.ai.track.model.TrackEventType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class RoiCounterRuleTest {

	private static RoiCounterRule rule() {
		return new RoiCounterRule("zone", Roi.rectangle(0f, 0f, 100f, 100f));
	}

	@Test
	void enterAndLeaveAreCountedExactlyOnce() {
		RoiCounterRule rule = rule();
		Track track = TrackFixture.track(1, 50f, 50f, 5);

		List<TrackEvent> entered = TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		assertThat(entered).hasSize(1);
		assertThat(entered.get(0).getType()).isEqualTo(TrackEventType.ROI_ENTERED);
		assertThat(entered.get(0).getDetail()).contains("firstSeenInside=true");
		assertThat(rule.getEnterCount()).isEqualTo(1);
		assertThat(rule.getCurrentInside()).containsExactly(1);

		TrackFixture.move(track, 1, 100L, 150f, 50f);
		List<TrackEvent> left = TrackFixture.run(rule, TrackFixture.frame(track, 1, 100L));
		assertThat(left).hasSize(1);
		assertThat(left.get(0).getType()).isEqualTo(TrackEventType.ROI_LEFT);
		assertThat(rule.getLeaveCount()).isEqualTo(1);
		assertThat(rule.getCurrentInside()).isEmpty();

		TrackFixture.move(track, 2, 200L, 160f, 50f);
		assertThat(TrackFixture.run(rule, TrackFixture.frame(track, 2, 200L))).isEmpty();
		assertThat(rule.getLeaveCount()).isEqualTo(1);
	}

	@Test
	void vanishedTrackEmitsCompensatingLeave() {
		RoiCounterRule rule = rule();
		Track track = TrackFixture.track(1, 50f, 50f, 5);
		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));

		List<TrackEvent> events = TrackFixture.run(rule, TrackFixture.emptyFrame(1, 100L));
		assertThat(events).hasSize(1);
		assertThat(events.get(0).getType()).isEqualTo(TrackEventType.ROI_LEFT);
		assertThat(events.get(0).getDetail()).contains("trackLost=true");
		assertThat(rule.getLeaveCount()).isEqualTo(1);
		assertThat(rule.getCurrentInside()).isEmpty();
	}

	@Test
	void anchorPolicyChangesTheDecision() {
		Roi roi = Roi.rectangle(0f, 50f, 100f, 60f);
		BBox box = new BBox(45f, 43f, 55f, 53f, 0.9f, 3);
		Track track = new Track(1, box, 5, 0, 64, 0, 0L);
		FrameContext context = TrackFixture.frame(track, 0, 0L);

		RoiCounterRule center = new RoiCounterRule("center", roi, AnchorPolicy.CENTER, true);
		RoiCounterRule bottom = new RoiCounterRule("bottom", roi, AnchorPolicy.BOTTOM_CENTER, true);
		assertThat(TrackFixture.run(center, context)).isEmpty();
		assertThat(TrackFixture.run(bottom, context)).hasSize(1);
		assertThat(bottom.getEnterCount()).isEqualTo(1);
	}

	@Test
	void polygonRoiAndReset() {
		Roi roi = Roi.polygon(0f, 0f, 200f, 0f, 200f, 200f, 0f, 200f);
		assertThat(roi.vertexCount()).isEqualTo(4);
		assertThat(roi.contains(50f, 50f)).isTrue();
		assertThat(roi.contains(250f, 250f)).isFalse();

		RoiCounterRule rule = new RoiCounterRule("poly", roi);
		Track track = TrackFixture.track(1, 50f, 50f, 5);
		TrackFixture.run(rule, TrackFixture.frame(track, 0, 0L));
		assertThat(rule.getEnterCount()).isEqualTo(1);

		rule.reset();
		assertThat(rule.getEnterCount()).isZero();
		assertThat(rule.getLeaveCount()).isZero();
		assertThat(rule.getCurrentInside()).isEmpty();
	}

	@Test
	void invalidRoiFailsFast() {
		assertThatThrownBy(() -> Roi.polygon(0f, 0f, 1f, 1f))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("3 个顶点");
		assertThatThrownBy(() -> new RoiCounterRule("x", Roi.rectangle(0f, 0f, 10f, 10f),
			null, true))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("anchor");
	}
}
