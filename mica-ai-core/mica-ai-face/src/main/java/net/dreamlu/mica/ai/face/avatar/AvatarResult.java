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
package net.dreamlu.mica.ai.face.avatar;

import lombok.Data;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;

/**
 * 头像提取结果。
 *
 * <p>字段含义：
 * <ul>
 *   <li>{@code image}：{@code size × size} 的 BGR Mat，调用方负责 {@link #release()}</li>
 *   <li>{@code box}：原图坐标系下使用的最终人脸框（可能因 {@code autoOrient} 重检测）</li>
 *   <li>{@code windowQuad}：裁剪窗口 4 角点（原图坐标系），便于上层二次裁剪</li>
 *   <li>{@code size}：输出头像边长（像素）</li>
 *   <li>{@code faceSize}：窗口内人脸占边长比例（诊断用）</li>
 *   <li>{@code orientationDegrees}：旋转角度（0/90/180/270）</li>
 *   <li>{@code tiledDetection}：是否走了 tile 分块检测</li>
 *   <li>{@code sharpness}：拉普拉斯方差，越大越清晰</li>
 *   <li>{@code usable} / {@code unusableReason}：业务可用性 + 原因</li>
 *   <li>{@code index}：在多脸场景下被选中的候选序号</li>
 * </ul>
 */
@Data
public class AvatarResult {

	private Mat image;
	private FaceBox box;
	private float[][] windowQuad;
	private int size;
	private double faceSize;
	private int orientationDegrees;
	private boolean tiledDetection;
	private double sharpness;
	private boolean usable;
	private String unusableReason;
	private int index;

	/**
	 * 释放 {@link #image} 持有的 native 资源；其它字段无 native 句柄。
	 * 调用方在不再使用本结果时必须调用本方法（推荐 try-with-resources 或显式 release）。
	 */
	public void release() {
		ImageUtils.releaseAll(image);
	}
}