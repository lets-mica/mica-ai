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

import net.dreamlu.mica.ai.track.config.ByteTrackConfig;
import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link ByteTracker} 主流程端到端单测。
 *
 * <p>全部用合成 BBox 序列驱动，零 ONNX / 零 OpenCV / 零图像。
 */
class ByteTrackerTest {

	private ByteTracker tracker;

	@BeforeEach
	void setUp() {
		tracker = ByteTracker.create(ByteTrackConfig.builder()
			.highThreshold(0.6f)
			.lowThreshold(0.1f)
			.matchThresh(0.8f)
			.matchLowThresh(0.5f)
			.trackBufferedThresh(5) // 故意调小加速删除
			.confirmAge(2)
			.build());
	}

	private static List<BBox> list(BBox... boxes) {
		List<BBox> l = new ArrayList<>();
		Collections.addAll(l, boxes);
		return l;
	}

	private static BBox bbox(float x1, float y1, float x2, float y2, float score) {
		return new BBox(x1, y1, x2, y2, score);
	}

	// ==== 配置校验 ====

	@Test
	void nullConfigThrows() {
		assertThatThrownBy(() -> ByteTracker.create(null))
			.isInstanceOf(IllegalArgumentException.class);
	}

	@Test
	void invertedThresholdsRejected() {
		assertThatThrownBy(() -> ByteTrackConfig.builder()
			.highThreshold(0.1f).lowThreshold(0.6f).build().validate())
			.isInstanceOf(IllegalArgumentException.class);
	}

	// ==== 帧 0：空输入 ====

	@Test
	void emptyInputKeepsState() {
		List<Track> r = tracker.update(Collections.<BBox>emptyList());
		assertThat(r).isEmpty();
		assertThat(tracker.trackCount()).isEqualTo(0);
	}

	// ==== 新建轨迹 ====

	@Test
	void firstDetCreatesNewTrack() {
		List<Track> r = tracker.update(list(bbox(0f, 0f, 50f, 50f, 0.9f)));
		assertThat(r).hasSize(1);
		Track t = r.get(0);
		assertThat(t.getId()).isEqualTo(1);
		assertThat(t.getAge()).isEqualTo(1);
		assertThat(t.isMature(2)).isFalse(); // confirmAge=2，age=1 还不成熟
	}

	// ==== 持续跟踪 + 成熟 ====

	@Test
	void consecutiveFramesKeepIdAndMature() {
		// 帧 0：新建（age=1，不成熟）
		tracker.update(list(bbox(0f, 0f, 50f, 50f, 0.9f)));
		// 帧 1：det IoU 很高 → 第一轮匹配（age=2，成熟）
		List<Track> r1 = tracker.update(list(bbox(1f, 1f, 51f, 51f, 0.9f)));
		assertThat(r1).hasSize(1);
		assertThat(r1.get(0).getId()).isEqualTo(1);
		assertThat(r1.get(0).isMature(2)).isTrue();
		// 帧 2：继续
		List<Track> r2 = tracker.update(list(bbox(2f, 2f, 52f, 52f, 0.9f)));
		assertThat(r2).hasSize(1);
		assertThat(r2.get(0).getId()).isEqualTo(1);
	}

	// ==== 多目标同时跟踪 ====

	@Test
	void twoDistinctTargetsKeepDistinctIds() {
		List<Track> r0 = tracker.update(list(
			bbox(0f, 0f, 50f, 50f, 0.9f),
			bbox(200f, 200f, 250f, 250f, 0.9f)));
		assertThat(r0).hasSize(2);
		assertThat(r0.get(0).getId()).isNotEqualTo(r0.get(1).getId());

		// 帧 1：各自小幅移动，应 ID 稳定
		List<Track> r1 = tracker.update(list(
			bbox(1f, 1f, 51f, 51f, 0.9f),
			bbox(201f, 201f, 251f, 251f, 0.9f)));
		assertThat(r1).hasSize(2);
	}

	// ==== 轨迹消失 → 删除 ====

	@Test
	void missingFramesEventuallyDeleteTrack() {
		// 先建立一个成熟轨迹
		tracker.update(list(bbox(0f, 0f, 50f, 50f, 0.9f)));
		tracker.update(list(bbox(1f, 1f, 51f, 51f, 0.9f)));
		assertThat(tracker.trackCount()).isEqualTo(1);

		// 连续 6 帧（> trackBufferedThresh=5）空 → 应该被删
		for (int i = 0; i < 6; i++) {
			tracker.update(Collections.<BBox>emptyList());
		}
		assertThat(tracker.trackCount()).isEqualTo(0);
	}

	// ==== ByteTrack 灵魂：低分二次匹配 ====

	@Test
	void lowScoreDetKeepsAliveTrack() {
		// 先建立成熟轨迹
		tracker.update(list(bbox(0f, 0f, 50f, 50f, 0.9f)));
		tracker.update(list(bbox(1f, 1f, 51f, 51f, 0.9f)));

		// 帧 2：同一位置，但 score 降到 0.2（在 lowThreshold=0.1 和 highThreshold=0.6 之间）
		// 第一轮高分匹配拿不到它 → 走第二轮低分匹配
		List<Track> r = tracker.update(list(bbox(2f, 2f, 52f, 52f, 0.2f)));
		assertThat(r).hasSize(1);
		assertThat(r.get(0).getId()).isEqualTo(1);
	}

	@Test
	void veryLowScoreDetGetsDropped() {
		// 先建立成熟轨迹
		tracker.update(list(bbox(0f, 0f, 50f, 50f, 0.9f)));
		tracker.update(list(bbox(1f, 1f, 51f, 51f, 0.9f)));

		// score=0.05 低于 lowThreshold=0.1 → 直接丢弃（连第二轮都进不去）
		// 已有成熟轨迹没被任何 det 匹配 → onUnmatched（lostFrames+1），但还没到删除阈值
		List<Track> r = tracker.update(list(bbox(2f, 2f, 52f, 52f, 0.05f)));
		assertThat(r).hasSize(1);
		assertThat(r.get(0).getId()).isEqualTo(1);
		// 再等 5 帧空（lostFrames 已 1，再加 5 变成 6 > 5）→ 删除
		for (int i = 0; i < 5; i++) {
			tracker.update(Collections.<BBox>emptyList());
		}
		assertThat(tracker.trackCount()).isEqualTo(0);
	}

	// ==== reset ====

	@Test
	void resetClearsAllState() {
		tracker.update(list(bbox(0f, 0f, 50f, 50f, 0.9f)));
		tracker.update(list(bbox(1f, 1f, 51f, 51f, 0.9f)));
		assertThat(tracker.trackCount()).isEqualTo(1);
		tracker.reset();
		assertThat(tracker.trackCount()).isEqualTo(0);
	}

	// ==== 新生轨迹 + 消失 → 再出现视为新目标 ====

	@Test
	void reappearanceGetsNewIdAfterDelete() {
		tracker.update(list(bbox(0f, 0f, 50f, 50f, 0.9f)));
		// 空帧超过缓冲 → 删除
		for (int i = 0; i < 6; i++) {
			tracker.update(Collections.<BBox>emptyList());
		}
		assertThat(tracker.trackCount()).isEqualTo(0);

		// 同一位置再出现 → 新 ID（id=2）
		List<Track> r = tracker.update(list(bbox(0f, 0f, 50f, 50f, 0.9f)));
		assertThat(r).hasSize(1);
		assertThat(r.get(0).getId()).isEqualTo(2);
	}

	// ==== 测试配置默认值 ====

	@Test
	void defaultConfigReasonable() {
		ByteTrackConfig c = ByteTrackConfig.builder().build();
		assertThat(c.getHighThreshold()).isEqualTo(0.6f);
		assertThat(c.getLowThreshold()).isEqualTo(0.1f);
		assertThat(c.getConfirmAge()).isEqualTo(3);
		assertThat(c.getTrackBufferedThresh()).isEqualTo(30);
	}
}
