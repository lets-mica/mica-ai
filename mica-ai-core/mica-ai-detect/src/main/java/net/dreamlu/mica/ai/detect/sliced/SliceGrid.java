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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 切片网格：把一张大图按固定边长 + 固定步进切成若干正方形切片。
 *
 * <p><b>切片边长必须等于模型输入边长</b>。原因：现有 letterbox 的缩放比是
 * {@code ratio = min(size / w, size / h)}，只有当切片是正方形且边长 == 模型输入边长时
 * {@code ratio} 才恒为 1、无填充，每片送进模型的预处理契约才与整图推理完全一致，
 * 坐标反算才退化为「加切片原点」。切片边长若小于模型输入边长，letterbox 会把切片
 * <b>放大</b>，小目标优势直接归零。
 *
 * <p><b>末列 / 末行回退</b>：起点按步进递增，当最后一片的右边界超出图像时，
 * 把最后一个起点回退到 {@code 边界 - 边长}，保证图像边缘被完整覆盖且不越界。
 *
 * <p>本类为纯几何计算，无状态、线程安全。
 */
public final class SliceGrid {

	private final int width;
	private final int height;
	private final int size;
	private final int step;
	private final List<Slice> slices;

	private SliceGrid(int width, int height, int size, int step, List<Slice> slices) {
		this.width = width;
		this.height = height;
		this.size = size;
		this.step = step;
		this.slices = Collections.unmodifiableList(slices);
	}

	/**
	 * 计算切片网格。
	 *
	 * @param width  原图宽（像素，正数）
	 * @param height 原图高（像素，正数）
	 * @param size   切片边长（正数，等于模型输入边长）
	 * @param step   步进（正数，须小于 {@code size} 才有重叠）
	 * @return 切片网格
	 * @throws IllegalArgumentException 任一参数非法
	 */
	public static SliceGrid of(int width, int height, int size, int step) {
		if (width <= 0 || height <= 0) {
			throw new IllegalArgumentException("width / height 必须为正数");
		}
		if (size <= 0) {
			throw new IllegalArgumentException("size 必须为正数");
		}
		if (step <= 0) {
			throw new IllegalArgumentException("step 必须为正数");
		}
		List<Slice> slices = new ArrayList<>();
		for (int y : starts(height, size, step)) {
			for (int x : starts(width, size, step)) {
				slices.add(new Slice(x, y, size));
			}
		}
		return new SliceGrid(width, height, size, step, slices);
	}

	/**
	 * 一维方向上的切片起点序列：递增步进，末位回退到「边界 - 边长」。
	 */
	private static List<Integer> starts(int boundary, int size, int step) {
		List<Integer> starts = new ArrayList<>();
		int pos = 0;
		while (pos + size < boundary) {
			starts.add(pos);
			pos += step;
		}
		int last = boundary - size;
		if (last < 0) {
			last = 0;
		}
		if (starts.isEmpty() || starts.get(starts.size() - 1) != last) {
			starts.add(last);
		}
		return starts;
	}

	/**
	 * 切片列表（行优先：先横向再纵向）。
	 *
	 * @return 不可修改的切片列表
	 */
	public List<Slice> getSlices() {
		return slices;
	}

	/**
	 * 切片总数。
	 *
	 * @return 切片数
	 */
	public int getSliceCount() {
		return slices.size();
	}

	/**
	 * 原图宽。
	 *
	 * @return 宽（像素）
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * 原图高。
	 *
	 * @return 高（像素）
	 */
	public int getHeight() {
		return height;
	}

	/**
	 * 切片边长。
	 *
	 * @return 边长（像素）
	 */
	public int getSize() {
		return size;
	}

	/**
	 * 切片步进。
	 *
	 * @return 步进（像素）
	 */
	public int getStep() {
		return step;
	}

}
