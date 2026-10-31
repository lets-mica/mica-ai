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
package net.dreamlu.mica.ai.docalign.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString;

/**
 * 文档角点（原图坐标系）。
 *
 * <p>{@link #score} 是该角点所在热力图通道的最大概率：
 * 正常命中时通常 &gt; 0.5；模型找不到该角点时回退到「热力图最大值位置」，
 * 此时 score 可能较低，调用方可据此判断是否需要人工复核。
 */
@Getter
@ToString
@AllArgsConstructor
public class DocCorner {

	private final double x;
	private final double y;
	private final double score;

	/**
	 * 到另一个角点的欧氏距离。
	 *
	 * @param other 另一个角点
	 * @return 距离（像素）
	 */
	public double distanceTo(DocCorner other) {
		double dx = x - other.x;
		double dy = y - other.y;
		return Math.sqrt(dx * dx + dy * dy);
	}

}
