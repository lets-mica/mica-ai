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
package net.dreamlu.mica.ai.track.core;

import net.dreamlu.mica.ai.track.algo.KalmanFilter;
import net.dreamlu.mica.ai.track.config.ByteTrackConfig;
import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;
import net.dreamlu.mica.ai.track.model.TrackPoint;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class ByteTrackerTimestampTest {

	private static ByteTrackConfig config(int historySize) {
		return ByteTrackConfig.builder()
			.highThreshold(0.5f)
			.lowThreshold(0.01f)
			.matchThresh(0.1f)
			.matchLowThresh(0.1f)
			.confirmAge(1)
			.trackBufferedThresh(5)
			.historySize(historySize)
			.build();
	}

	private static BBox box(float x) {
		return new BBox(x, 100f, x + 20f, 120f, 0.9f, 3);
	}

	private static List<BBox> one(float x) {
		return Collections.singletonList(box(x));
	}

	@Test
	void timestampedUpdateRecordsPointsVelocityAndDirection() {
		ByteTracker tracker = ByteTracker.create(config(32));

		tracker.update(one(100f), 1000L);
		List<Track> tracks = tracker.update(one(110f), 1500L);

		assertThat(tracks).hasSize(1);
		Track track = tracks.get(0);
		assertThat(tracker.getFrameIndex()).isEqualTo(2);
		assertThat(track.getFirstSeenMs()).isEqualTo(1000L);
		assertThat(track.getLastSeenMs()).isEqualTo(1500L);
		assertThat(track.getFirstFrameIndex()).isEqualTo(1);
		assertThat(track.getLastFrameIndex()).isEqualTo(2);
		assertThat(track.dwellMs()).isEqualTo(500L);
		assertThat(track.getPoints()).hasSize(2);
		assertThat(track.getPoints().get(0).getTimestampMs()).isEqualTo(1000L);
		assertThat(track.getPoints().get(1).getTimestampMs()).isEqualTo(1500L);
		assertThat(track.velocityXPerSecond()).isCloseTo(20d, org.assertj.core.data.Offset.offset(0.01d));
		assertThat(track.velocityYPerSecond()).isCloseTo(0d, org.assertj.core.data.Offset.offset(0.01d));
		assertThat(track.directionDegrees()).isCloseTo(0d, org.assertj.core.data.Offset.offset(0.01d));
		assertThat(track.directionLabel()).isEqualTo("右");
		assertThat(track.getLastBBox().getX1()).isEqualTo(110f);
		assertThat(track.getLastBBox().getClassId()).isEqualTo(3);
	}

	@Test
	void legacyUpdateKeepsNoTimestampAndDoesNotFakeVelocity() {
		ByteTracker tracker = ByteTracker.create(config(32));

		tracker.update(one(100f));
		List<Track> tracks = tracker.update(one(110f));

		Track track = tracks.get(0);
		assertThat(track.getLastPoint().getTimestampMs()).isEqualTo(Track.NO_TIMESTAMP);
		assertThat(track.dwellMs()).isEqualTo(-1L);
		assertThat(track.velocityXPerSecond()).isEqualTo(0d);
		assertThat(track.speedPerSecond()).isEqualTo(0d);
		assertThat(track.directionLabel()).isEqualTo("静止");
		assertThat(tracker.getFrameIndex()).isEqualTo(2);
	}

	@Test
	void historySizeCapsPointsAndObservedPointsFilterPredictions() {
		ByteTracker tracker = ByteTracker.create(config(4));

		for (int i = 1; i <= 6; i++) {
			tracker.update(one(100f + i * 2f), 1000L + i * 100L);
		}
		List<Track> tracks = tracker.update(Collections.<BBox>emptyList(), 1700L);
		assertThat(tracks).hasSize(1);

		Track track = tracks.get(0);
		assertThat(track.getPoints()).hasSize(4);
		assertThat(track.getPoints().get(0).getFrameIndex()).isEqualTo(4);
		assertThat(track.getPoints().get(3).getFrameIndex()).isEqualTo(7);
		assertThat(track.getPoints().get(3).isObserved()).isFalse();
		assertThat(track.observedPoints(10)).hasSize(3);
		for (TrackPoint point : track.observedPoints(10)) {
			assertThat(point.isObserved()).isTrue();
		}
	}

	@Test
	void resetClearsFrameIndexAndTrackState() {
		ByteTracker tracker = ByteTracker.create(config(32));
		tracker.update(one(100f), 1000L);
		tracker.update(one(102f), 1100L);
		assertThat(tracker.getFrameIndex()).isEqualTo(2);

		tracker.reset();
		assertThat(tracker.getFrameIndex()).isZero();
		assertThat(tracker.trackCount()).isZero();

		List<Track> tracks = tracker.update(one(100f), 2000L);
		assertThat(tracker.getFrameIndex()).isEqualTo(1);
		assertThat(tracks.get(0).getId()).isEqualTo(1);
		assertThat(tracks.get(0).getFirstSeenMs()).isEqualTo(2000L);
	}

	@Test
	void invalidConfigurationsFailFast() {
		assertThatThrownBy(() -> ByteTrackConfig.builder().historySize(2).build().validate())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("historySize");
		assertThatThrownBy(() -> ByteTracker.create(config(32)).update(one(100f), -2L))
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("timestampMs");
	}

	@Test
	void kalmanExposesPerFrameVelocity() {
		KalmanFilter kf = new KalmanFilter(new BBox(0f, 0f, 20f, 20f, 0.9f));
		assertThat(kf.velocityX()).isEqualTo(0d);
		assertThat(kf.velocityY()).isEqualTo(0d);
		kf.predict();
		assertThat(kf.velocityX()).isEqualTo(0d);
		assertThat(kf.velocityY()).isEqualTo(0d);
	}
}
