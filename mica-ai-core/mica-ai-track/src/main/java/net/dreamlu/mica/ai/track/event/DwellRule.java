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
import net.dreamlu.mica.ai.track.model.TrackPoint;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 滞留判定：轨迹锚点在 ROI 内持续停留超过 {@code thresholdMs} 毫秒时触发一次事件。
 *
 * <p>用真实采集时间戳而不是帧数：掉帧、变速播放、抽帧检测都不会把「帧数」误当「时长」。
 * 因此本规则**必须**使用带时间戳的 {@code update(List<BBox>, long)} / {@code process(dets, timestampMs)}，
 * 否则直接 fail-fast。
 *
 * <p>离开 ROI 后清除状态，再次进入可以重新计时并再次上报。
 */
public class DwellRule implements TrackEventRule {

	private final String name;
	private final Roi roi;
	private final long thresholdMs;
	private final AnchorPolicy anchor;
	private final boolean requireMature;
	private final Map<Integer, Long> enterMs = new HashMap<>();
	private final Set<Integer> reported = new HashSet<>();

	public DwellRule(String name, Roi roi, long thresholdMs) {
		this(name, roi, thresholdMs, AnchorPolicy.CENTER, true);
	}

	public DwellRule(String name, Roi roi, long thresholdMs, AnchorPolicy anchor, boolean requireMature) {
		if (name == null || name.trim().isEmpty()) {
			throw new IllegalArgumentException("dwell name must not be blank");
		}
		if (roi == null || anchor == null) {
			throw new IllegalArgumentException("roi / anchor must not be null");
		}
		if (thresholdMs <= 0L) {
			throw new IllegalArgumentException("thresholdMs 必须为正整数");
		}
		this.name = name;
		this.roi = roi;
		this.thresholdMs = thresholdMs;
		this.anchor = anchor;
		this.requireMature = requireMature;
	}

	@Override
	public String name() {
		return name;
	}

	@Override
	public void onFrame(FrameContext context, List<TrackEvent> sink) {
		if (!context.hasTimestamp()) {
			throw new IllegalArgumentException(
				"DwellRule 需要真实时间戳：请用 ByteTracker.update(List<BBox>, long) 或 TrackEventEngine.process(dets, timestampMs)");
		}
		Set<Integer> alive = new HashSet<>();
		for (Track track : context.getTracks()) {
			alive.add(track.getId());
			if (requireMature && !track.isMature(context.getConfirmAge())) {
				continue;
			}
			TrackPoint point = track.getLastPoint();
			if (point == null || point.getFrameIndex() != context.getFrameIndex()) {
				continue;
			}
			BBox box = track.getLastBBox();
			boolean inside = roi.contains(anchor.anchorX(box), anchor.anchorY(box));
			if (!inside) {
				enterMs.remove(track.getId());
				reported.remove(track.getId());
				continue;
			}
			Long enter = enterMs.get(track.getId());
			if (enter == null) {
				enterMs.put(track.getId(), context.getTimestampMs());
				continue;
			}
			long stayMs = context.getTimestampMs() - enter;
			if (stayMs >= thresholdMs && !reported.contains(track.getId())) {
				reported.add(track.getId());
				sink.add(new TrackEvent(TrackEventType.DWELL_EXCEEDED, track.getId(),
					track.getClassId(), context.getFrameIndex(), context.getTimestampMs(),
					name, "", "zone=" + name + ", stayMs=" + stayMs + ", thresholdMs=" + thresholdMs, box));
			}
		}
		enterMs.keySet().retainAll(alive);
		reported.retainAll(alive);
	}

	@Override
	public void reset() {
		enterMs.clear();
		reported.clear();
	}
}
