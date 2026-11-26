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
package net.dreamlu.mica.ai.track.motion;

import net.dreamlu.mica.ai.track.config.ByteTrackConfig;
import net.dreamlu.mica.ai.track.core.ByteTracker;
import net.dreamlu.mica.ai.track.event.RoiCounterRule;
import net.dreamlu.mica.ai.track.event.Roi;
import net.dreamlu.mica.ai.track.event.AnchorPolicy;
import net.dreamlu.mica.ai.track.event.TrackEventEngine;
import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;
import net.dreamlu.mica.ai.track.model.TrackPoint;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 相机运动补偿测试（零 ONNX / OpenCV 依赖）。
 *
 * <p>场景：目标在世界坐标里静止，相机向右摇 → 图像内容整体左移 {@code dx} 像素/帧。
 * 检测框在图像坐标系里逐帧左移，在<b>参考帧</b>（相机固定系）里保持不动。
 *
 * <p>无补偿时轨迹框与检测框逐帧错位，IoU 掉到匹配阈值以下 → 轨迹断裂、ID 换新；
 * 有补偿时检测框被变换回参考帧再参与匹配 → 轨迹 ID 稳定。
 */
class MotionCompensationTest {

	private static final float BOX_SIZE = 40f;
	private static final float START_X = 200f;
	private static final float START_Y = 200f;
	private static final float DX = 20f;
	private static final int FRAMES = 5;
	private static final long FRAME_INTERVAL_MS = 100L;

	private static ByteTrackConfig config() {
		return ByteTrackConfig.builder()
			.highThreshold(0.5f)
			.lowThreshold(0.1f)
			.matchThresh(0.8f)
			.matchLowThresh(0.5f)
			.confirmAge(2)
			.trackBufferedThresh(30)
			.historySize(32)
			.build();
	}

	/**
	 * 第 {@code frame} 帧的检测框（图像坐标系）：目标在世界静止，图像内容整体左移 dx。
	 */
	private static BBox detectionAt(int frame) {
		float x = START_X - DX * frame;
		return new BBox(x, START_Y, x + BOX_SIZE, START_Y + BOX_SIZE, 0.9f, 0);
	}

	/**
	 * 推进相机运动模型：第 0 帧即<b>参考帧</b>（不追加增量），之后每帧追加一次「相对上一帧」的增量。
	 */
	private static void advance(AffineMotionModel motion, int frame) {
		if (motion != null && frame > 0) {
			motion.append(AffineMotionModel.translation(-DX, 0f));
		}
	}

	/**
	 * 跑一串帧，返回出现过的全部轨迹 id。
	 *
	 * @param motion 相机运动模型，可为 null（无补偿）
	 */
	private static Set<Integer> runFrames(ByteTracker tracker, AffineMotionModel motion) {
		Set<Integer> ids = new LinkedHashSet<>();
		for (int frame = 0; frame < FRAMES; frame++) {
			advance(motion, frame);
			List<Track> tracks = tracker.update(Collections.singletonList(detectionAt(frame)),
				frame * FRAME_INTERVAL_MS);
			for (Track track : tracks) {
				ids.add(track.getId());
			}
		}
		return ids;
	}

	@Test
	void withoutCompensationTrackIdBreaks() {
		ByteTracker tracker = ByteTracker.create(config());

		Set<Integer> ids = runFrames(tracker, null);

		// 相机一动，轨迹框与检测框错位，IoU 掉到 0.8 阈值以下 → 每帧都换新 ID
		assertThat(ids).hasSizeGreaterThan(1);
	}

	@Test
	void withCompensationTrackIdStaysStable() {
		AffineMotionModel motion = AffineMotionModel.identity();
		ByteTracker tracker = ByteTracker.create(config(), motion);

		Set<Integer> ids = runFrames(tracker, motion);

		// 检测框被变换回参考帧后再匹配 → 目标在参考帧里静止，轨迹不断裂
		assertThat(ids).hasSize(1);
	}

	@Test
	void compensatedVelocityReflectsWorldMotionNotApparentMotion() {
		AffineMotionModel motion = AffineMotionModel.identity();
		ByteTracker tracker = ByteTracker.create(config(), motion);

		List<Track> last = new ArrayList<>();
		for (int frame = 0; frame < FRAMES; frame++) {
			advance(motion, frame);
			last = tracker.update(Collections.singletonList(detectionAt(frame)),
				frame * FRAME_INTERVAL_MS);
		}

		Track track = last.get(0);
		// 图像坐标下目标逐帧左移 20px（表观运动）
		assertThat(track.getLastPoint().getCx()).isLessThan(START_X);
		// 参考帧坐标下目标静止（真实运动）→ 速度为 0
		assertThat(Math.abs(track.velocityXPerSecond())).isLessThan(1e-6d);
	}

	@Test
	void trajectoryPointsKeepImageCoordinatesAndCarryReferencePosition() {
		AffineMotionModel motion = AffineMotionModel.identity();
		ByteTracker tracker = ByteTracker.create(config(), motion);

		List<Track> last = new ArrayList<>();
		for (int frame = 0; frame < FRAMES; frame++) {
			advance(motion, frame);
			last = tracker.update(Collections.singletonList(detectionAt(frame)),
				frame * FRAME_INTERVAL_MS);
		}

		List<TrackPoint> points = last.get(0).getPoints();
		assertThat(points).isNotEmpty();
		for (TrackPoint point : points) {
			// 轨迹点仍在图像坐标系（ROI / 越线判定用）
			assertThat(point.hasReference()).isTrue();
			// 参考帧坐标恒定（目标在世界静止）
			assertThat(point.getRefCx()).isEqualTo(START_X + BOX_SIZE / 2f);
		}
	}

	@Test
	void withoutCompensationTrackPointsCarryNoReferencePosition() {
		ByteTracker tracker = ByteTracker.create(config());

		List<Track> last = new ArrayList<>();
		for (int frame = 0; frame < FRAMES; frame++) {
			last = tracker.update(Collections.singletonList(detectionAt(frame)),
				frame * FRAME_INTERVAL_MS);
		}

		// 未启用补偿 → 没有「参考帧」这回事，参考帧坐标必须为 null（与历史版本逐位一致）
		for (TrackPoint point : last.get(0).getPoints()) {
			assertThat(point.hasReference()).isFalse();
			assertThat(point.getRefCx()).isNull();
			assertThat(point.getRefCy()).isNull();
		}
	}

	@Test
	void roiCounterIsStableWithCompensation() {
		AffineMotionModel motion = AffineMotionModel.identity();
		ByteTracker tracker = ByteTracker.create(config(), motion);
		Roi zone = Roi.rectangle(START_X - 200f, START_Y - 100f, START_X + 200f, START_Y + 100f);
		RoiCounterRule counter = new RoiCounterRule("zone", zone, AnchorPolicy.CENTER, false);
		TrackEventEngine engine = TrackEventEngine.create(tracker, counter);

		for (int frame = 0; frame < FRAMES; frame++) {
			advance(motion, frame);
			engine.process(Collections.singletonList(detectionAt(frame)), frame * FRAME_INTERVAL_MS);
		}

		// 目标始终在 ROI 内 → 只进一次、不会因为 ID 换新而重复计数
		assertThat(counter.getEnterCount()).isEqualTo(1);
		assertThat(counter.getCurrentInside()).hasSize(1);
	}

}
