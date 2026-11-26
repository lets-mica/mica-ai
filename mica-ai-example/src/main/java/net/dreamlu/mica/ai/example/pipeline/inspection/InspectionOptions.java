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

import lombok.Data;

import java.util.ArrayList;
import java.util.List;

/**
 * 巡检管线的阶段开关与规则参数。
 */
@Data
public class InspectionOptions {

	/** 是否用切片检测（关闭则整图检测） */
	private boolean sliced = true;

	/** 是否启用相机运动补偿 */
	private boolean motionCompensation = true;

	/**
	 * 每帧相对<b>上一帧</b>的相机运动增量（仿射系数 {@code [a,b,c,d,e,f]}）。
	 *
	 * <p>第 1 帧即参考帧，不追加增量；长度 = 帧数 - 1。
	 */
	private List<float[]> motionIncrements = new ArrayList<>();

	/** 判定线端点 {@code [x1,y1,x2,y2]} */
	private float[] line = {100f, 0f, 100f, 400f};

	/** ROI 矩形 {@code [x1,y1,x2,y2]} */
	private float[] roi = {0f, 0f, 200f, 200f};

	/** ROI 锚点是否取底边中心（行人 / 车辆推荐 true） */
	private boolean bottomCenterAnchor = true;

	/** 滞留阈值（毫秒） */
	private long dwellThresholdMs = 1000L;

	/** 帧间隔（毫秒），用于合成采集时间戳 */
	private long frameIntervalMs = 100L;

}
