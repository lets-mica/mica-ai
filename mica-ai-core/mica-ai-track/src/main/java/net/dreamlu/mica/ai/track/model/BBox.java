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

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 检测器吐出的单帧检测框（x1, y1, x2, y2 左上角 / 右下角，score 置信度）。
 *
 * <p>不可变。
 */
@Getter
@AllArgsConstructor
public class BBox {

	private final float x1;
	private final float y1;
	private final float x2;
	private final float y2;
	private final float score;
	private final int classId;

	public BBox(float x1, float y1, float x2, float y2, float score) {
		this(x1, y1, x2, y2, score, 0);
	}

	public float cx() {
		return (x1 + x2) * 0.5f;
	}

	public float cy() {
		return (y1 + y2) * 0.5f;
	}

	public float width() {
		return x2 - x1;
	}

	public float height() {
		return y2 - y1;
	}

	public boolean isEmpty() {
		return width() <= 0 || height() <= 0;
	}

	/**
	 * 与另一个框的 IoU（交并比）。空框或无交集时返回 0。
	 */
	public float iou(BBox other) {
		if (isEmpty() || other.isEmpty()) {
			return 0f;
		}
		float ix1 = Math.max(this.x1, other.x1);
		float iy1 = Math.max(this.y1, other.y1);
		float ix2 = Math.min(this.x2, other.x2);
		float iy2 = Math.min(this.y2, other.y2);
		float inter = Math.max(0f, ix2 - ix1) * Math.max(0f, iy2 - iy1);
		if (inter <= 0f) {
			return 0f;
		}
		float u = this.width() * this.height() + other.width() * other.height() - inter;
		if (u <= 0f) {
			return 0f;
		}
		return inter / u;
	}
}
