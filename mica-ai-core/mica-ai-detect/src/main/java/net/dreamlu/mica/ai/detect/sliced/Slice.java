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
package net.dreamlu.mica.ai.detect.sliced;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * 切片推理的一个切片（正方形），值为其在**原图**坐标系下的左上角。
 *
 * <p>切片边长恒等于模型输入边长（见 {@link SliceGrid}），因此每片送进模型的
 * 预处理契约与整图推理完全一致（letterbox 缩放比为 1、无填充）。
 */
@Getter
@RequiredArgsConstructor
public class Slice {

	/** 切片左上角 x（原图坐标） */
	private final int x;

	/** 切片左上角 y（原图坐标） */
	private final int y;

	/** 切片边长（正方形） */
	private final int size;

	/**
	 * 切片右下角 x（原图坐标，可能超出原图边界，由调用方补齐）。
	 */
	public int getX2() {
		return x + size;
	}

	/**
	 * 切片右下角 y（原图坐标，可能超出原图边界，由调用方补齐）。
	 */
	public int getY2() {
		return y + size;
	}

	/**
	 * 本切片与给定矩形的交集（像素）。
	 *
	 * @param otherX1 矩形左上角 x
	 * @param otherY1 矩形左上角 y
	 * @param otherX2 矩形右下角 x
	 * @param otherY2 矩形右下角 y
	 * @return 交集面积；不相交时为 0
	 */
	public long overlapArea(int otherX1, int otherY1, int otherX2, int otherY2) {
		int ix1 = Math.max(x, otherX1);
		int iy1 = Math.max(y, otherY1);
		int ix2 = Math.min(getX2(), otherX2);
		int iy2 = Math.min(getY2(), otherY2);
		if (ix2 <= ix1 || iy2 <= iy1) {
			return 0L;
		}
		return (long) (ix2 - ix1) * (iy2 - iy1);
	}

}
