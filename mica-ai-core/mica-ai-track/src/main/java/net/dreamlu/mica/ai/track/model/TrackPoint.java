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

import lombok.Getter;
import lombok.ToString;

/**
 * 轨迹上的单个采样点（不可变）。
 *
 * <p>{@code observed=true} 表示该帧有实测检测框命中；{@code false} 表示只有 Kalman 预测框。
 * 行为规则默认只用实测点，避免遮挡期的预测框造成误报。
 *
 * <p>{@code cx/cy} 与 {@code x1/y1/x2/y2} 是<b>当前帧</b>（图像坐标系）下的位置 ——
 * ROI / 越线这类按画面位置判定的规则用它。
 *
 * <p>{@code refCx/refCy} 是<b>参考帧</b>（相机固定坐标系）下的中心点，仅在启用相机运动补偿时
 * 才有值；速度与方向用它计算，得到的是目标在世界里的真实运动，而不是「相机运动 + 目标运动」
 * 的表观运动。
 */
@Getter
@ToString
public class TrackPoint {

	private final int frameIndex;
	private final long timestampMs;
	private final float cx;
	private final float cy;
	private final float x1;
	private final float y1;
	private final float x2;
	private final float y2;
	private final boolean observed;
	/** 参考帧中心 x；未启用相机运动补偿时为 null */
	private final Float refCx;
	/** 参考帧中心 y；未启用相机运动补偿时为 null */
	private final Float refCy;

	/**
	 * 构造轨迹点（无参考帧坐标）。
	 *
	 * @param frameIndex 帧序号
	 * @param timestampMs 采集时间戳（毫秒），无时间信息时为 -1
	 * @param cx         中心 x
	 * @param cy         中心 y
	 * @param x1         左上角 x
	 * @param y1         左上角 y
	 * @param x2         右下角 x
	 * @param y2         右下角 y
	 * @param observed   是否为实测点
	 */
	public TrackPoint(int frameIndex, long timestampMs, float cx, float cy,
					  float x1, float y1, float x2, float y2, boolean observed) {
		this(frameIndex, timestampMs, cx, cy, x1, y1, x2, y2, observed, null, null);
	}

	/**
	 * 构造轨迹点（带参考帧坐标）。
	 *
	 * @param frameIndex 帧序号
	 * @param timestampMs 采集时间戳（毫秒），无时间信息时为 -1
	 * @param cx         当前帧中心 x
	 * @param cy         当前帧中心 y
	 * @param x1         左上角 x
	 * @param y1         左上角 y
	 * @param x2         右下角 x
	 * @param y2         右下角 y
	 * @param observed   是否为实测点
	 * @param refCx      参考帧中心 x，可为 null
	 * @param refCy      参考帧中心 y，可为 null
	 */
	public TrackPoint(int frameIndex, long timestampMs, float cx, float cy,
					  float x1, float y1, float x2, float y2, boolean observed,
					  Float refCx, Float refCy) {
		this.frameIndex = frameIndex;
		this.timestampMs = timestampMs;
		this.cx = cx;
		this.cy = cy;
		this.x1 = x1;
		this.y1 = y1;
		this.x2 = x2;
		this.y2 = y2;
		this.observed = observed;
		this.refCx = refCx;
		this.refCy = refCy;
	}

	public boolean hasTimestamp() {
		return timestampMs >= 0L;
	}

	/**
	 * 是否带参考帧坐标。
	 *
	 * @return 带参考帧坐标返回 true
	 */
	public boolean hasReference() {
		return refCx != null && refCy != null;
	}
}
