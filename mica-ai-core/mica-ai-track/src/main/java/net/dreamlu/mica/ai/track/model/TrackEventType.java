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

/**
 * 轨迹事件类型。
 */
public enum TrackEventType {

	/** 轨迹中心点穿过判定线 */
	LINE_CROSSED,
	/** 轨迹进入 ROI */
	ROI_ENTERED,
	/** 轨迹离开 ROI */
	ROI_LEFT,
	/** 在 ROI 内驻留超过阈值 */
	DWELL_EXCEEDED,
	/** 运动方向发生大幅折返 */
	DIRECTION_CHANGED
}
