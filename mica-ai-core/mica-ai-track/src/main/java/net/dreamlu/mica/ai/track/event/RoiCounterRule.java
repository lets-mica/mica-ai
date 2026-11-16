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

import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * ROI 进出计数：按锚点判断是否落入多边形区域，跨边界时产生进入 / 离开事件。
 *
 * <p>轨迹被 ByteTracker 删除时，若此前仍记为「在场」，会补发一次 {@code ROI_LEFT}，
 * 避免人走出画面导致计数只增不减。
 */
public class RoiCounterRule implements TrackEventRule {

	private static final class Presence {
		private boolean inside;
		private BBox box;
		private int classId;
	}

	private final String name;
	private final Roi roi;
	private final AnchorPolicy anchor;
	private final boolean requireMature;
	private final Map<Integer, Presence> presence = new HashMap<>();
	private int enterCount;
	private int leaveCount;

	public RoiCounterRule(String name, Roi roi) {
		this(name, roi, AnchorPolicy.CENTER, true);
	}

	public RoiCounterRule(String name, Roi roi, AnchorPolicy anchor, boolean requireMature) {
		if (name == null || name.trim().isEmpty()) {
			throw new IllegalArgumentException("roi name must not be blank");
		}
		if (roi == null || anchor == null) {
			throw new IllegalArgumentException("roi / anchor must not be null");
		}
		this.name = name;
		this.roi = roi;
		this.anchor = anchor;
		this.requireMature = requireMature;
	}

	@Override
	public String name() {
		return name;
	}

	@Override
	public void onFrame(FrameContext context, List<TrackEvent> sink) {
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
			boolean nowInside = roi.contains(anchor.anchorX(box), anchor.anchorY(box));
			Presence state = presence.get(track.getId());
			if (state == null) {
				state = new Presence();
				state.inside = nowInside;
				presence.put(track.getId(), state);
				if (nowInside) {
					enterCount++;
					emit(context, sink, TrackEventType.ROI_ENTERED, track, box,
						"firstSeenInside=true, enters=" + enterCount);
				}
			} else if (state.inside != nowInside) {
				state.inside = nowInside;
				if (nowInside) {
					enterCount++;
					emit(context, sink, TrackEventType.ROI_ENTERED, track, box,
						"enters=" + enterCount);
				} else {
					leaveCount++;
					emit(context, sink, TrackEventType.ROI_LEFT, track, box,
						"leaves=" + leaveCount);
				}
			}
			state.box = box;
			state.classId = track.getClassId();
		}

		Iterator<Map.Entry<Integer, Presence>> it = presence.entrySet().iterator();
		while (it.hasNext()) {
			Map.Entry<Integer, Presence> entry = it.next();
			if (!alive.contains(entry.getKey())) {
				Presence state = entry.getValue();
				if (state.inside) {
					leaveCount++;
					BBox box = state.box == null ? new BBox(0f, 0f, 0f, 0f, 0f, state.classId) : state.box;
					sink.add(new TrackEvent(TrackEventType.ROI_LEFT, entry.getKey(),
						state.classId, context.getFrameIndex(), context.getTimestampMs(),
						name, "", "trackLost=true, leaves=" + leaveCount, box));
				}
				it.remove();
			}
		}
	}

	private void emit(FrameContext context, List<TrackEvent> sink, TrackEventType type,
					  Track track, BBox box, String detail) {
		sink.add(new TrackEvent(type, track.getId(), track.getClassId(),
			context.getFrameIndex(), context.getTimestampMs(), name, "", detail, box));
	}

	@Override
	public void reset() {
		presence.clear();
		enterCount = 0;
		leaveCount = 0;
	}

	public int getEnterCount() {
		return enterCount;
	}

	public int getLeaveCount() {
		return leaveCount;
	}

	public Set<Integer> getCurrentInside() {
		Set<Integer> result = new HashSet<>();
		for (Map.Entry<Integer, Presence> entry : presence.entrySet()) {
			if (entry.getValue().inside) {
				result.add(entry.getKey());
			}
		}
		return Collections.unmodifiableSet(result);
	}
}
