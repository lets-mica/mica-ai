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
package net.dreamlu.mica.ai.track.motion;

import net.dreamlu.mica.ai.track.model.BBox;

/**
 * 相机运动模型：在**参考帧**（相机固定坐标系）与**当前帧**（图像坐标系）之间互转。
 *
 * <p><b>为什么需要它</b>：球机云台转动、车载摄像机移动时，静止目标的框在图像里整体位移，
 * 轨迹框与检测框的 IoU 掉到匹配阈值以下 → 轨迹断裂、ID 换新 → 越线 / ROI 进出 /
 * 滞留 / 折返事件全部失真。
 *
 * <p><b>参考帧</b>是跟踪器创建那一刻的图像坐标系。跟踪器内部（Kalman 状态、匹配代价矩阵）
 * 一律在参考帧下计算：每帧的检测框先经 {@link #toReference(BBox)} 变换回参考帧再参与匹配，
 * 于是<b>相机运动对匹配完全没有影响</b> —— 这比「把轨迹框往前推一帧」更彻底：
 * 后者在相机匀速平移时会把相机运动重复计入（Kalman 速度本身就是图像坐标下的速度，
 * 已经吸收了一部分相机运动）。
 *
 * <p><b>由调用方维护并逐帧更新</b>：参考帧 → 当前帧的变换是<b>累积</b>的
 * （第 t 帧的变换 = 第 1..t 帧逐帧变换的复合）。实现见 {@link AffineMotionModel#compose}。
 * 参考帧（模型状态 = 恒等的那一帧）不追加增量，从第 2 帧起每帧追加一次「相对上一帧」的增量。
 */
public interface MotionModel {

	/**
	 * 参考帧 → 当前帧。
	 *
	 * @param referenceBox 参考帧坐标系下的框
	 * @return 当前帧坐标系下的框
	 */
	BBox toCurrent(BBox referenceBox);

	/**
	 * 当前帧 → 参考帧。
	 *
	 * @param currentBox 当前帧坐标系下的框
	 * @return 参考帧坐标系下的框
	 */
	BBox toReference(BBox currentBox);

}
