/*
 * Copyright (c) 2019-2029, DreamLu 卢春梦 (596392912@qq.com & dreamlu.net).
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
package net.dreamlu.mica.ai.example.pipeline.inspection;

import lombok.Getter;
import net.dreamlu.mica.ai.track.model.TrackEvent;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * 巡检管线结果：逐帧轨迹 + 事件流 + 汇总 + 最后一帧标注图。
 */
@Getter
public class InspectionResult {

	/** 逐帧报告 */
	private final List<FrameReport> frames;

	/** 全部事件（按帧序） */
	private final List<TrackEvent> events;

	/** 汇总（各事件类型计数 + enters / leaves / maxInside / trackIdCount） */
	private final Map<String, Object> summary;

	/** 最后一帧的标注 PNG；无帧时为 null */
	private final byte[] annotatedPng;

	public InspectionResult(List<FrameReport> frames, List<TrackEvent> events,
							 Map<String, Object> summary, byte[] annotatedPng) {
		this.frames = Collections.unmodifiableList(frames);
		this.events = Collections.unmodifiableList(events);
		this.summary = Collections.unmodifiableMap(summary);
		this.annotatedPng = annotatedPng;
	}

	/**
	 * 单帧报告。
	 */
	@Getter
	public static class FrameReport {

		/** 帧序号（从 1 开始） */
		private final int frameIndex;
		/** 采集时间戳（毫秒） */
		private final long timestampMs;
		/** 本帧切片数（整图模式为 1） */
		private final int sliceCount;
		/** 本帧检出的目标数 */
		private final int detectCount;
		/** 本帧轨迹 */
		private final List<TrackView> tracks;
		/** 本帧事件 */
		private final List<TrackEvent> events;

		public FrameReport(int frameIndex, long timestampMs, int sliceCount, int detectCount,
						   List<TrackView> tracks, List<TrackEvent> events) {
			this.frameIndex = frameIndex;
			this.timestampMs = timestampMs;
			this.sliceCount = sliceCount;
			this.detectCount = detectCount;
			this.tracks = Collections.unmodifiableList(tracks);
			this.events = Collections.unmodifiableList(events);
		}
	}

	/**
	 * 轨迹视图：图像坐标 + 参考帧坐标 + 速度方向。
	 */
	@Getter
	public static class TrackView {

		/** 轨迹 id */
		private final int id;
		/** 类别 id */
		private final int classId;
		/** 类别名（检测模型标签集） */
		private final String label;
		/** 置信度 */
		private final float score;
		/** 当前帧中心 x（图像坐标） */
		private final float cx;
		/** 当前帧中心 y（图像坐标） */
		private final float cy;
		/** 参考帧中心 x（相机固定系）；未启用补偿时为 null */
		private final Float refCx;
		/** 参考帧中心 y（相机固定系）；未启用补偿时为 null */
		private final Float refCy;
		/** 速度（像素 / 秒，参考帧口径） */
		private final double speedPxPerSec;
		/** 方向标签 */
		private final String direction;
		/** 累计命中帧数 */
		private final int age;

		public TrackView(int id, int classId, String label, float score, float cx, float cy,
						  Float refCx, Float refCy, double speedPxPerSec, String direction, int age) {
			this.id = id;
			this.classId = classId;
			this.label = label;
			this.score = score;
			this.cx = cx;
			this.cy = cy;
			this.refCx = refCx;
			this.refCy = refCy;
			this.speedPxPerSec = speedPxPerSec;
			this.direction = direction;
			this.age = age;
		}
	}

}
