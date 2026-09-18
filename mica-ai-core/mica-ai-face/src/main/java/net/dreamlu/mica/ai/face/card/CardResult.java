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
package net.dreamlu.mica.ai.face.card;

import lombok.Data;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;

/**
 * 证件卡片提取结果。
 *
 * <p>字段含义：
 * <ul>
 *   <li>{@code image}：矫正后 BGR Mat（{@code CardOptions#getOutputWidth()}
 *       × {@code getOutputHeight()}），调用方负责 {@link #release()}</li>
 *   <li>{@code quad}：检测到的 4 角点（原图坐标系）</li>
 *   <li>{@code aspectRatio}：长宽比（宽 / 高）</li>
 *   <li>{@code score}：四边形拟合得分（0~1）</li>
 *   <li>{@code rotationDegrees}：旋转角度（0/90/180/270）</li>
 *   <li>{@code autoOriented}：是否经过自动摆正</li>
 *   <li>{@code cardSize}：矫正后卡面占边长比例（诊断用）</li>
 *   <li>{@code sharpness}：拉普拉斯方差，越大越清晰</li>
 *   <li>{@code usable} / {@code unusableReason}：业务可用性 + 原因</li>
 *   <li>{@code index}：在多候选场景下被选中的候选序号</li>
 * </ul>
 */
@Data
public class CardResult {

	private Mat image;
	private float[][] quad;
	private double aspectRatio;
	private double score;
	private int rotationDegrees;
	private boolean autoOriented;
	private double cardSize;
	private double sharpness;
	private boolean usable;
	private String unusableReason;
	private int index;

	/**
	 * 释放 {@link #image} 持有的 native 资源；其它字段无 native 句柄。
	 */
	public void release() {
		ImageUtils.releaseAll(image);
	}
}