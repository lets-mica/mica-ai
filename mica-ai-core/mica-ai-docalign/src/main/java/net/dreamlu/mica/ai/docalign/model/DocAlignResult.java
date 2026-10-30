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

import lombok.Getter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 文档角点检测结果。
 *
 * <p>{@link #getCorners()} 固定 4 个点，顺序与上游 DocAligner 一致且已实测确认：
 * <b>左上 → 右上 → 右下 → 左下</b>（即 "Z" 字形顺序，逆时针绕行）。
 */
@Getter
public class DocAlignResult {

	public static final int CORNER_COUNT = 4;

	private final List<DocCorner> corners;
	private final int width;
	private final int height;

	/**
	 * 构造结果。
	 *
	 * @param corners 4 个角点（左上 / 右上 / 右下 / 左下）
	 * @param width   原图宽度
	 * @param height  原图高度
	 */
	public DocAlignResult(List<DocCorner> corners, int width, int height) {
		if (corners == null || corners.size() != CORNER_COUNT) {
			throw new IllegalArgumentException("corners 必须为 4 个角点");
		}
		this.corners = Collections.unmodifiableList(new ArrayList<>(corners));
		this.width = width;
		this.height = height;
	}

	/**
	 * 左上角点。
	 *
	 * @return 左上角点
	 */
	public DocCorner getTopLeft() {
		return corners.get(0);
	}

	/**
	 * 右上角点。
	 *
	 * @return 右上角点
	 */
	public DocCorner getTopRight() {
		return corners.get(1);
	}

	/**
	 * 右下角点。
	 *
	 * @return 右下角点
	 */
	public DocCorner getBottomRight() {
		return corners.get(2);
	}

	/**
	 * 左下角点。
	 *
	 * @return 左下角点
	 */
	public DocCorner getBottomLeft() {
		return corners.get(3);
	}

	@Override
	public String toString() {
		return "DocAlignResult{corners=" + corners + ", size=" + width + "x" + height + '}';
	}
}
