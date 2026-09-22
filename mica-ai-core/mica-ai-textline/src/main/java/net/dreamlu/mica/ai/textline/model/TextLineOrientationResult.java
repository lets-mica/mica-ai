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
package net.dreamlu.mica.ai.textline.model;

import lombok.Getter;

/**
 * 文本行方向分类结果。
 *
 * <p>{@link #orientation} 是最终判定方向，{@link #score} 是其置信度（softmax 概率）。
 * {@link #rawLogits} 保留模型原始输出，便于排查阈值问题或换模型时对照。
 *
 * <p>本结果不含任何 OpenCV {@code Mat}（不持有原生内存），因此<b>无需 close</b>，
 * 可自由缓存与传递。
 */
@Getter
public class TextLineOrientationResult {

	private final TextLineOrientation orientation;
	private final float score;
	private final float[] rawLogits;

	public TextLineOrientationResult(TextLineOrientation orientation, float score, float[] rawLogits) {
		this.orientation = orientation;
		this.score = score;
		this.rawLogits = rawLogits;
	}

	/**
	 * 是否判定为倒置（需要旋转 180 度转正）。
	 *
	 * @return {@code 180 度} 时返回 {@code true}
	 */
	public boolean isUpsideDown() {
		return orientation != null && orientation.isUpsideDown();
	}

	/**
	 * 把该行转正所需旋转角度。
	 *
	 * @return 0 或 180；方向为 null 时返回 0
	 */
	public int angle() {
		return orientation == null ? 0 : orientation.getAngle();
	}

	@Override
	public String toString() {
		return "TextLineOrientationResult{orientation=" + orientation
			+ ", score=" + score + '}';
	}
}
