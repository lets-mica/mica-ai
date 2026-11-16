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
import net.dreamlu.mica.ai.track.config.ByteTrackConfig;
import net.dreamlu.mica.ai.track.core.ByteTracker;
import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;
import net.dreamlu.mica.ai.track.model.TrackEvent;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 逐帧推理管线的纯算法部分：检测框 + 采集时间戳 → 稳定轨迹 + 行为事件。
 *
 * <p>本类刻意**不碰图像、不碰视频解码**：调用方（例如 mica-video 的帧回调）负责把
 * {@code Mat} / {@code Frame} 交给检测器，再把 {@code DetectResult} 转成 {@code List<BBox>} 喂进来。
 * 这样 mica-ai-track 保持零模型、零 ONNX、零 OpenCV、零 Spring 依赖。
 *
 * <p><b>有状态且非线程安全</b>：一路视频流必须独占一个 {@code TrackEventEngine}
 * （内部 ByteTracker 与每条规则都保存跨帧状态），换流时调 {@link #reset()}。
 * 不实现 {@code AutoCloseable}——没有可释放资源。
 *
 * <p>时间戳要求单调不减；滞留类规则还要求时间戳非负，否则会直接抛异常而不是按帧数猜时长。
 */
@Getter
public class TrackEventEngine {

	private final ByteTracker tracker;
	private final List<TrackEventRule> rules;
	private long lastTimestampMs = -1L;

	public TrackEventEngine(ByteTracker tracker, List<TrackEventRule> rules) {
		if (tracker == null) {
			throw new IllegalArgumentException("tracker must not be null");
		}
		if (rules == null) {
			throw new IllegalArgumentException("rules must not be null");
		}
		List<TrackEventRule> copy = new ArrayList<>(rules);
		for (TrackEventRule rule : copy) {
			if (rule == null) {
				throw new IllegalArgumentException("rules 中不能包含 null");
			}
		}
		this.tracker = tracker;
		this.rules = Collections.unmodifiableList(copy);
	}

	public static TrackEventEngine create(ByteTrackConfig config, TrackEventRule... rules) {
		return new TrackEventEngine(new ByteTracker(config), Arrays.asList(rules));
	}

	public static TrackEventEngine create(ByteTracker tracker, TrackEventRule... rules) {
		return new TrackEventEngine(tracker, Arrays.asList(rules));
	}

	/**
	 * 无时间戳重载：轨迹点 {@code timestampMs = -1}，滞留类规则不可用。
	 */
	public FrameResult process(List<BBox> detections) {
		return process(detections, Track.NO_TIMESTAMP);
	}

	/**
	 * 处理一帧。
	 *
	 * @param detections  本帧检测框，可为 {@code null} / 空（表示本帧无目标）
	 * @param timestampMs 本帧采集时间戳（毫秒，单调不减）；无时间信息时传 -1
	 */
	public FrameResult process(List<BBox> detections, long timestampMs) {
		if (timestampMs < Track.NO_TIMESTAMP) {
			throw new IllegalArgumentException("timestampMs 必须是 -1 或非负数");
		}
		if (timestampMs >= 0L) {
			if (lastTimestampMs >= 0L && timestampMs < lastTimestampMs) {
				throw new IllegalArgumentException("timestampMs 必须单调不减，上一帧=" + lastTimestampMs
					+ "，本帧=" + timestampMs);
			}
			lastTimestampMs = timestampMs;
		}
		List<Track> tracks = tracker.update(detections, timestampMs);
		FrameContext context = new FrameContext(tracker.getFrameIndex(), timestampMs,
			tracks, tracker.getConfig().getConfirmAge());
		List<TrackEvent> events = new ArrayList<>();
		for (TrackEventRule rule : rules) {
			rule.onFrame(context, events);
		}
		return new FrameResult(context.getFrameIndex(), timestampMs, tracks, events);
	}

	/**
	 * 清空跟踪器与全部规则状态（换视频流 / 场景切换时用）。
	 */
	public void reset() {
		tracker.reset();
		for (TrackEventRule rule : rules) {
			rule.reset();
		}
		lastTimestampMs = -1L;
	}
}
