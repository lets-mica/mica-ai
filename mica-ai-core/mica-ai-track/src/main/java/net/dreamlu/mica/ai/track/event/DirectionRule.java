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
import net.dreamlu.mica.ai.track.util.Directions;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * 折返 / 逆行判定：连续三段实测位移的夹角超过阈值时触发一次方向突变事件。
 *
 * <p>只取实测点（不吃 Kalman 预测点），并要求两段位移都大于 {@code minDisplacement}，
 * 避免静止抖动被当成折返；同一轨迹在 {@code cooldownFrames} 帧内不重复上报。
 */
public class DirectionRule implements TrackEventRule {

	private final String name;
	private final double turnAngleDegrees;
	private final float minDisplacement;
	private final int cooldownFrames;
	private final boolean requireMature;
	private final Map<Integer, Integer> lastReportedFrame = new HashMap<>();

	public DirectionRule(String name) {
		this(name, 120d, 5f, 10, true);
	}

	public DirectionRule(String name, double turnAngleDegrees, float minDisplacement,
						 int cooldownFrames, boolean requireMature) {
		if (name == null || name.trim().isEmpty()) {
			throw new IllegalArgumentException("direction name must not be blank");
		}
		if (turnAngleDegrees <= 0d || turnAngleDegrees > 180d) {
			throw new IllegalArgumentException("turnAngleDegrees 必须在 (0, 180]");
		}
		if (minDisplacement <= 0f) {
			throw new IllegalArgumentException("minDisplacement 必须为正数");
		}
		if (cooldownFrames < 0) {
			throw new IllegalArgumentException("cooldownFrames 不能为负数");
		}
		this.name = name;
		this.turnAngleDegrees = turnAngleDegrees;
		this.minDisplacement = minDisplacement;
		this.cooldownFrames = cooldownFrames;
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
			List<TrackPoint> points = track.observedPoints(3);
			if (points.size() < 3) {
				continue;
			}
			TrackPoint p0 = points.get(0);
			TrackPoint p1 = points.get(1);
			TrackPoint p2 = points.get(2);
			if (p2.getFrameIndex() != context.getFrameIndex()) {
				continue;
			}
			double v1x = p1.getCx() - p0.getCx();
			double v1y = p1.getCy() - p0.getCy();
			double v2x = p2.getCx() - p1.getCx();
			double v2y = p2.getCy() - p1.getCy();
			double len1 = Math.sqrt(v1x * v1x + v1y * v1y);
			double len2 = Math.sqrt(v2x * v2x + v2y * v2y);
			if (len1 < minDisplacement || len2 < minDisplacement) {
				continue;
			}
			double cos = (v1x * v2x + v1y * v2y) / (len1 * len2);
			cos = Math.max(-1d, Math.min(1d, cos));
			double angle = Math.toDegrees(Math.acos(cos));
			if (angle < turnAngleDegrees) {
				continue;
			}
			Integer lastFrame = lastReportedFrame.get(track.getId());
			if (lastFrame != null && context.getFrameIndex() - lastFrame < cooldownFrames) {
				continue;
			}
			lastReportedFrame.put(track.getId(), context.getFrameIndex());
			String from = Directions.label(v1x, v1y);
			String to = Directions.label(v2x, v2y);
			BBox box = track.getLastBBox();
			sink.add(new TrackEvent(TrackEventType.DIRECTION_CHANGED, track.getId(),
				track.getClassId(), context.getFrameIndex(), context.getTimestampMs(),
				name, to, "turnAngle=" + String.format(Locale.ROOT, "%.1f", angle)
					+ ", from=" + from + ", to=" + to, box));
		}
		lastReportedFrame.keySet().retainAll(alive);
	}

	@Override
	public void reset() {
		lastReportedFrame.clear();
	}
}
