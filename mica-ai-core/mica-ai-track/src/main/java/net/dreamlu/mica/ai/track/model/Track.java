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
import net.dreamlu.mica.ai.track.algo.KalmanFilter;

/**
 * 内部轨迹状态。
 */
@Getter
public class Track {

	private final int id;
	private final KalmanFilter kf;
	private float score;
	private int classId;
	/** 自上次被匹配以来经过的帧数；0 表示刚刚匹配成功 */
	private int lostFrames;
	/** 累计匹配成功的帧数；达到 confirmAge 前不算成熟轨迹 */
	private int age;
	/** 轨迹结束时刻（被删除时设） */
	private boolean finished;

	public Track(int id, BBox bbox, int age, int lostFrames) {
		this.id = id;
		this.kf = new KalmanFilter(bbox);
		this.score = bbox.getScore();
		this.classId = bbox.getClassId();
		this.age = age;
		this.lostFrames = lostFrames;
	}

	public BBox getCurrentBBox() {
		BBox pred = kf.predictedBBox();
		if (lostFrames == 0) {
			return pred;
		}
		return pred;
	}

	public void onMatched(BBox bbox) {
		this.kf.update(bbox);
		this.score = bbox.getScore();
		this.classId = bbox.getClassId();
		this.lostFrames = 0;
		this.age++;
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
}
