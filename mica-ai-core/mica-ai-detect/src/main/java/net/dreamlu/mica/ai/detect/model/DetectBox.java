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
package net.dreamlu.mica.ai.detect.model;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * 单个检测框（原图坐标系，{@code x1/y1/x2/y2} 为绝对像素，左上 / 右下）。
 */
@Getter
@ToString
@RequiredArgsConstructor
public class DetectBox {

	private final float x1;
	private final float y1;
	private final float x2;
	private final float y2;
	private final float score;
	private final int classId;

	/**
	 * 检测框宽度。
	 *
	 * @return {@code x2 - x1}
	 */
	public float width() {
		return x2 - x1;
	}

	/**
	 * 检测框高度。
	 *
	 * @return {@code y2 - y1}
	 */
	public float height() {
		return y2 - y1;
	}

	/**
	 * 检测框面积。
	 *
	 * @return 宽 × 高
	 */
	public float area() {
		return width() * height();
	}

	/**
	 * 计算与另一个检测框的 IoU。
	 *
	 * @param other 另一个检测框
	 * @return 交并比，无交集时为 0
	 */
	public float iou(DetectBox other) {
		float ix1 = Math.max(x1, other.x1);
		float iy1 = Math.max(y1, other.y1);
		float ix2 = Math.min(x2, other.x2);
		float iy2 = Math.min(y2, other.y2);
		float iw = ix2 - ix1;
		float ih = iy2 - iy1;
		if (iw <= 0f || ih <= 0f) {
			return 0f;
		}
		float inter = iw * ih;
		float union = area() + other.area() - inter;
		return union <= 0f ? 0f : inter / union;
	}

	/**
	 * 按类别名表解析类别名（越界时返回 {@code classId} 字符串）。
	 *
	 * @param labels 类别名表，可为 {@code null}
	 * @return 类别名
	 */
	public String label(String[] labels) {
		if (labels == null || classId < 0 || classId >= labels.length) {
			return String.valueOf(classId);
		}
		return labels[classId];
	}
}
