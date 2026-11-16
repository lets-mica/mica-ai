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
package net.dreamlu.mica.ai.track.model;

import lombok.AccessLevel;
import lombok.Getter;
import net.dreamlu.mica.ai.track.algo.KalmanFilter;
import net.dreamlu.mica.ai.track.util.Directions;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * 内部轨迹状态。
 *
 * <p>除 ByteTrack 原有的 id / 年龄 / 丢失帧数外，额外保留有限长度的轨迹点历史、
 * 首末见时间与帧序号，供越线、ROI 计数、滞留、方向等纯几何行为规则使用。
 * 速度按「相邻实测点位移 ÷ 真实时间差」计算，掉帧时也不会把帧数错当时间。
 */
@Getter
public class Track {

	/** 未提供采集时间戳时使用的哨兵值。 */
	public static final long NO_TIMESTAMP = -1L;

	private static final int DEFAULT_HISTORY_SIZE = 32;

	private final int id;
	private final KalmanFilter kf;
	private final int historySize;
	@Getter(AccessLevel.NONE)
	private final ArrayDeque<TrackPoint> points;
	private float score;
	private int classId;
	/** 自上次被匹配以来经过的帧数；0 表示刚刚匹配成功 */
	private int lostFrames;
	/** 累计匹配成功的帧数；达到 confirmAge 前不算成熟轨迹 */
	private int age;
	/** 轨迹结束时刻（被删除时设） */
	private boolean finished;
	private BBox lastMatchedBBox;
	private long firstSeenMs;
	private long lastSeenMs;
	private int firstFrameIndex;
	private int lastFrameIndex;

	public Track(int id, BBox bbox, int age, int lostFrames) {
		this(id, bbox, age, lostFrames, DEFAULT_HISTORY_SIZE, 0, NO_TIMESTAMP);
	}

	public Track(int id, BBox bbox, int age, int lostFrames, int historySize, int frameIndex, long timestampMs) {
		if (bbox == null) {
			throw new IllegalArgumentException("bbox must not be null");
		}
		if (historySize <= 0) {
			throw new IllegalArgumentException("historySize 必须为正整数");
		}
		if (timestampMs < NO_TIMESTAMP) {
			throw new IllegalArgumentException("timestampMs 必须是 -1 或非负数");
		}
		this.id = id;
		this.kf = new KalmanFilter(bbox);
		this.historySize = historySize;
		this.points = new ArrayDeque<>(historySize);
		this.score = bbox.getScore();
		this.classId = bbox.getClassId();
		this.age = age;
		this.lostFrames = lostFrames;
		this.lastMatchedBBox = bbox;
		this.firstSeenMs = timestampMs;
		this.lastSeenMs = timestampMs;
		this.firstFrameIndex = frameIndex;
		this.lastFrameIndex = frameIndex;
		appendPoint(frameIndex, timestampMs, bbox, true);
	}

	public BBox getCurrentBBox() {
		return kf.predictedBBox();
	}

	public void onMatched(BBox bbox) {
		this.kf.update(bbox);
		this.score = bbox.getScore();
		this.classId = bbox.getClassId();
		this.lostFrames = 0;
		this.age++;
		this.lastMatchedBBox = bbox;
	}

	public void onUnmatched() {
		this.lostFrames++;
	}

	public void markFinished() {
		this.finished = true;
	}

	public boolean isMature(int confirmAge) {
		return age >= confirmAge;
	}

	/**
	 * 追加一个轨迹点；超出 {@code historySize} 时丢弃最旧的点。
	 */
	public void appendPoint(int frameIndex, long timestampMs, BBox box, boolean observed) {
		while (points.size() >= historySize) {
			points.pollFirst();
		}
		points.addLast(new TrackPoint(frameIndex, timestampMs, box.cx(), box.cy(),
			box.getX1(), box.getY1(), box.getX2(), box.getY2(), observed));
		this.lastFrameIndex = frameIndex;
		this.lastSeenMs = timestampMs;
	}

	/**
	 * 轨迹点历史快照（时间正序，不可修改）。
	 */
	public List<TrackPoint> getPoints() {
		return Collections.unmodifiableList(new ArrayList<>(points));
	}

	public TrackPoint getLastPoint() {
		return points.peekLast();
	}

	/**
	 * 本帧采样点对应的框（实测框或 Kalman 预测框），行为规则与事件输出都用它，
	 * 保证「判定坐标」与「事件里的框」来自同一帧。
	 */
	public BBox getLastBBox() {
		TrackPoint last = points.peekLast();
		if (last == null) {
			return getCurrentBBox();
		}
		return new BBox(last.getX1(), last.getY1(), last.getX2(), last.getY2(), score, classId);
	}

	/**
	 * 最近至多 {@code max} 个实测点（时间正序）。
	 */
	public List<TrackPoint> observedPoints(int max) {
		List<TrackPoint> result = new ArrayList<>();
		Iterator<TrackPoint> it = points.descendingIterator();
		while (it.hasNext() && result.size() < max) {
			TrackPoint p = it.next();
			if (p.isObserved()) {
				result.add(p);
			}
		}
		Collections.reverse(result);
		return result;
	}

	/**
	 * 轨迹存活时长（毫秒）；未提供时间戳时返回 -1。
	 */
	public long dwellMs() {
		return (firstSeenMs < 0L || lastSeenMs < 0L) ? -1L : lastSeenMs - firstSeenMs;
	}

	public double velocityXPerSecond() {
		TrackPoint[] pair = lastTwoTimedPoints();
		if (pair == null) {
			return 0d;
		}
		double dt = (pair[1].getTimestampMs() - pair[0].getTimestampMs()) / 1000d;
		return dt > 0d ? (pair[1].getCx() - pair[0].getCx()) / dt : 0d;
	}

	public double velocityYPerSecond() {
		TrackPoint[] pair = lastTwoTimedPoints();
		if (pair == null) {
			return 0d;
		}
		double dt = (pair[1].getTimestampMs() - pair[0].getTimestampMs()) / 1000d;
		return dt > 0d ? (pair[1].getCy() - pair[0].getCy()) / dt : 0d;
	}

	public double speedPerSecond() {
		double vx = velocityXPerSecond();
		double vy = velocityYPerSecond();
		return Math.sqrt(vx * vx + vy * vy);
	}

	/**
	 * 运动方向角度（度，屏幕坐标系：0° 向右，90° 向下，逆时针为负方向）。
	 */
	public double directionDegrees() {
		return Directions.degrees(velocityXPerSecond(), velocityYPerSecond());
	}

	/**
	 * 八方向中文标签；静止返回「静止」。
	 */
	public String directionLabel() {
		return Directions.label(velocityXPerSecond(), velocityYPerSecond());
	}

	private TrackPoint[] lastTwoTimedPoints() {
		TrackPoint newer = null;
		TrackPoint older = null;
		Iterator<TrackPoint> it = points.descendingIterator();
		while (it.hasNext()) {
			TrackPoint p = it.next();
			if (!p.hasTimestamp()) {
				continue;
			}
			if (newer == null) {
				newer = p;
			} else {
				older = p;
				break;
			}
		}
		return (newer == null || older == null) ? null : new TrackPoint[]{older, newer};
	}
}
