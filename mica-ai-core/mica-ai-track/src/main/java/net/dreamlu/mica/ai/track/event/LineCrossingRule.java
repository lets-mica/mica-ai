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
 * 越线判定：轨迹中心点从判定线的一侧移到另一侧时触发一次事件。
 *
 * <p>方向定义：以 A(x1,y1) → B(x2,y2) 为基准，叉积由负变正记为 {@code FORWARD}，
 * 反之 {@code BACKWARD}。可按 {@link Direction} 只统计其中一个方向（如「只算进场」）。
 *
 * <p>默认只用实测点、只统计成熟轨迹，避免遮挡期 Kalman 预测框造成误报。
 */
public class LineCrossingRule implements TrackEventRule {

	public static final String DIR_FORWARD = "FORWARD";
	public static final String DIR_BACKWARD = "BACKWARD";

	public enum Direction {
		BOTH, FORWARD, BACKWARD
	}

	private final String name;
	private final float ax;
	private final float ay;
	private final float bx;
	private final float by;
	private final Direction direction;
	private final boolean requireMature;
	private final boolean observedOnly;
	private final Map<Integer, Boolean> lastSide = new HashMap<>();

	public LineCrossingRule(String name, float x1, float y1, float x2, float y2) {
		this(name, x1, y1, x2, y2, Direction.BOTH, true, true);
	}

	public LineCrossingRule(String name, float x1, float y1, float x2, float y2,
							Direction direction, boolean requireMature, boolean observedOnly) {
		if (name == null || name.trim().isEmpty()) {
			throw new IllegalArgumentException("line name must not be blank");
		}
		if (x1 == x2 && y1 == y2) {
			throw new IllegalArgumentException("判定线两个端点不能重合");
		}
		if (direction == null) {
			throw new IllegalArgumentException("direction must not be null");
		}
		this.name = name;
		this.ax = x1;
		this.ay = y1;
		this.bx = x2;
		this.by = y2;
		this.direction = direction;
		this.requireMature = requireMature;
		this.observedOnly = observedOnly;
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
			if (observedOnly && !point.isObserved()) {
				continue;
			}
			double cross = (double) (bx - ax) * (point.getCy() - ay)
				- (double) (by - ay) * (point.getCx() - ax);
			if (Math.abs(cross) < 1e-6d) {
				continue;
			}
			boolean side = cross > 0d;
			Boolean previous = lastSide.get(track.getId());
			if (previous != null && previous != side) {
				String dir = side ? DIR_FORWARD : DIR_BACKWARD;
				boolean accepted = direction == Direction.BOTH
					|| (direction == Direction.FORWARD && side)
					|| (direction == Direction.BACKWARD && !side);
				if (accepted) {
					BBox box = track.getLastBBox();
					sink.add(new TrackEvent(TrackEventType.LINE_CROSSED, track.getId(),
						track.getClassId(), context.getFrameIndex(), context.getTimestampMs(),
						name, dir, "line=" + name + ", direction=" + dir, box));
				}
			}
			lastSide.put(track.getId(), side);
		}
		lastSide.keySet().retainAll(alive);
	}

	@Override
	public void reset() {
		lastSide.clear();
	}

	public int watchedTracks() {
		return lastSide.size();
	}
}
