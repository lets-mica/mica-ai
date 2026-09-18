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
package net.dreamlu.mica.ai.layout.postprocess;

import net.dreamlu.mica.ai.layout.config.LayoutConfig;
import net.dreamlu.mica.ai.layout.model.LayoutLabel;
import net.dreamlu.mica.ai.layout.model.LayoutResult;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * PP-DocLayoutV3 检测输出后处理。
 *
 * <p>模型输出 {@code [300, 7]}，每行为
 * {@code [class_id, score, x1, y1, x2, y2, order]}：**未做 NMS**（300 个 query 全输出），
 * 坐标为 **letterbox 画布坐标**（喂入 {@code scale_factor = 1 / letterboxScale} 时）；
 * 第 7 列是模型预测的阅读顺序键 —— PaddleX 对 7 列输出的处理就是
 * {@code np.argsort(boxes[:, 6])}（见 {@code LayoutAnalysisProcess.apply} 中
 * "boxes.shape[1] == 7 is new ordered object detection" 分支）。
 *
 * <p>处理链路对齐 PaddleX：
 * <ol>
 *   <li>反 letterbox {@code (out - pad) / letterboxScale} → 取整 → clip 到原图</li>
 *   <li>per-class 阈值过滤（{@link LayoutConfig#thresholdFor(int)}）</li>
 *   <li>NMS：同类 IoU &gt; {@code nmsThreshold}、异类 IoU &gt; {@code nmsDiffClassThreshold} 才互斥</li>
 *   <li>整页 {@code image} 伪框过滤（面积占比超 0.82 / 0.93 的整页图丢弃）</li>
 *   <li>{@code maxDetections} 截断 + reading order rank 解码</li>
 * </ol>
 */
public class LayoutPostProcessor {

	private static final int COL_CLASS = 0;
	private static final int COL_SCORE = 1;
	private static final int COL_X1 = 2;
	private static final int COL_Y1 = 3;
	private static final int COL_X2 = 4;
	private static final int COL_Y2 = 5;
	private static final int COL_ORDER = 6;
	private static final int MIN_COLUMNS = 6;
	private static final int MIN_BOX_SIDE = 1;
	private static final float LARGE_IMAGE_AREA_RATIO_LANDSCAPE = 0.82f;
	private static final float LARGE_IMAGE_AREA_RATIO_PORTRAIT = 0.93f;

	private final LayoutConfig config;

	/**
	 * 构造后处理器。
	 *
	 * @param config 版面分析配置（阈值、NMS 等）
	 */
	public LayoutPostProcessor(LayoutConfig config) {
		this.config = config;
	}

	/**
	 * 把模型原始输出后处理为 {@link LayoutResult} 列表。
	 *
	 * <p>处理链路：反 letterbox → per-class 阈值 → NMS → 整页 {@code image} 伪框过滤 →
	 * {@code maxDetections} 截断 → reading order rank。
	 *
	 * @param boxes          模型原始输出 {@code [N, 7]}，letterbox 画布坐标
	 * @param letterboxScale letterbox 缩放比（= 输入边长 / 原图边长）
	 * @param padLeft        letterbox 左补边
	 * @param padTop         letterbox 上补边
	 * @param origW          原图宽
	 * @param origH          原图高
	 * @return 按 {@code LayoutResult#getReadingOrder()} 升序排列的版面区域列表
	 */
	public List<LayoutResult> postProcess(float[][] boxes, double letterboxScale,
										  int padLeft, int padTop,
										  int origW, int origH) {
		if (boxes == null || boxes.length == 0 || origW <= 0 || origH <= 0) {
			return LayoutResult.emptyList();
		}
		double scale = letterboxScale > 0 ? letterboxScale : 1d;
		List<RawBox> raws = new ArrayList<>(boxes.length);
		for (int i = 0; i < boxes.length; i++) {
			RawBox raw = toRawBox(boxes[i], i, scale, padLeft, padTop, origW, origH);
			if (raw != null) {
				raws.add(raw);
			}
		}
		raws.sort(new Comparator<RawBox>() {
			@Override
			public int compare(RawBox a, RawBox b) {
				return Float.compare(b.score, a.score);
			}
		});
		if (config.isLayoutNms() && raws.size() > 1) {
			raws = nms(raws);
		}
		float areaLimit = (origW > origH ? LARGE_IMAGE_AREA_RATIO_LANDSCAPE
			: LARGE_IMAGE_AREA_RATIO_PORTRAIT) * origW * origH;
		List<RawBox> kept = new ArrayList<>(raws.size());
		for (RawBox raw : raws) {
			if (isOverlargeImageBox(raw, areaLimit)) {
				continue;
			}
			kept.add(raw);
			if (kept.size() >= config.getMaxDetections()) {
				break;
			}
		}
		int[] readingOrder = decodeReadingOrder(kept);
		List<LayoutResult> out = new ArrayList<>(kept.size());
		for (int i = 0; i < kept.size(); i++) {
			RawBox raw = kept.get(i);
			out.add(new LayoutResult(raw.label, raw.label.getCode(), raw.box,
				raw.score, raw.index, i, readingOrder[i]));
		}
		return out;
	}

	private RawBox toRawBox(float[] row, int index, double scale,
							int padLeft, int padTop, int origW, int origH) {
		if (row == null || row.length < MIN_COLUMNS) {
			return null;
		}
		int clsId = Math.round(row[COL_CLASS]);
		if (clsId < 0) {
			return null;
		}
		LayoutLabel label = LayoutLabel.ofIndex(clsId);
		if (label == null) {
			return null;
		}
		float score = row[COL_SCORE];
		if (Float.isNaN(score) || score < config.thresholdFor(clsId)) {
			return null;
		}
		int x1 = toOrig(row[COL_X1], padLeft, scale, origW);
		int y1 = toOrig(row[COL_Y1], padTop, scale, origH);
		int x2 = toOrig(row[COL_X2], padLeft, scale, origW);
		int y2 = toOrig(row[COL_Y2], padTop, scale, origH);
		if (x1 > x2) {
			int tmp = x1;
			x1 = x2;
			x2 = tmp;
		}
		if (y1 > y2) {
			int tmp = y1;
			y1 = y2;
			y2 = tmp;
		}
		if (x2 - x1 < MIN_BOX_SIDE || y2 - y1 < MIN_BOX_SIDE) {
			return null;
		}
		float orderKey = row.length > COL_ORDER ? row[COL_ORDER] : Float.NaN;
		return new RawBox(new int[]{x1, y1, x2, y2}, score, label, index, orderKey);
	}

	private static int toOrig(float value, int pad, double scale, int limit) {
		int rounded = Math.round((float) ((value - pad) / scale));
		if (rounded < 0) {
			return 0;
		}
		return Math.min(rounded, limit);
	}

	/**
	 * 贪心 NMS：输入已按 score 降序，同类用 {@code nmsThreshold}、异类用
	 * {@code nmsDiffClassThreshold}（0.98 ⇒ 异类几乎不互斥，允许版面区域嵌套）。
	 */
	private List<RawBox> nms(List<RawBox> boxList) {
		int size = boxList.size();
		boolean[] removed = new boolean[size];
		List<RawBox> kept = new ArrayList<>(size);
		for (int i = 0; i < size; i++) {
			if (removed[i]) {
				continue;
			}
			RawBox current = boxList.get(i);
			kept.add(current);
			for (int j = i + 1; j < size; j++) {
				if (removed[j]) {
					continue;
				}
				RawBox other = boxList.get(j);
				float limit = current.label == other.label
					? config.getNmsThreshold() : config.getNmsDiffClassThreshold();
				if (iou(current.box, other.box) > limit) {
					removed[j] = true;
				}
			}
		}
		return kept;
	}

	static float iou(int[] a, int[] b) {
		int interW = Math.min(a[2], b[2]) - Math.max(a[0], b[0]);
		int interH = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
		if (interW <= 0 || interH <= 0) {
			return 0f;
		}
		float inter = (float) interW * interH;
		float areaA = (float) (a[2] - a[0]) * (a[3] - a[1]);
		float areaB = (float) (b[2] - b[0]) * (b[3] - b[1]);
		float union = areaA + areaB - inter;
		return union <= 0f ? 0f : inter / union;
	}

	private static boolean isOverlargeImageBox(RawBox raw, float areaLimit) {
		if (raw.label != LayoutLabel.IMAGE) {
			return false;
		}
		float area = (float) (raw.box[2] - raw.box[0]) * (raw.box[3] - raw.box[1]);
		return area > areaLimit;
	}

	/**
	 * 阅读顺序 rank：按模型第 7 列（order 键）升序，键相同则 score 高者先读。
	 * 对齐 PaddleX {@code np.argsort(boxes[:, 6])}；rank 0 = 最先读。
	 * 模型未输出该列时全部返回 {@link LayoutResult#READING_ORDER_NONE}。
	 */
	static int[] decodeReadingOrder(List<RawBox> boxList) {
		int size = boxList.size();
		int[] rank = new int[size];
		for (RawBox raw : boxList) {
			if (Float.isNaN(raw.orderKey) || Float.isInfinite(raw.orderKey)) {
				Arrays.fill(rank, LayoutResult.READING_ORDER_NONE);
				return rank;
			}
		}
		final List<RawBox> list = boxList;
		Integer[] idx = new Integer[size];
		for (int i = 0; i < size; i++) {
			idx[i] = i;
		}
		Arrays.sort(idx, new Comparator<Integer>() {
			@Override
			public int compare(Integer a, Integer b) {
				RawBox ra = list.get(a);
				RawBox rb = list.get(b);
				int byOrder = Float.compare(ra.orderKey, rb.orderKey);
				if (byOrder != 0) {
					return byOrder;
				}
				int byScore = Float.compare(rb.score, ra.score);
				return byScore != 0 ? byScore : Integer.compare(a, b);
			}
		});
		for (int pos = 0; pos < size; pos++) {
			rank[idx[pos]] = pos;
		}
		return rank;
	}

	static class RawBox {
		final int[] box;
		final float score;
		final LayoutLabel label;
		final int index;
		final float orderKey;

		RawBox(int[] box, float score, LayoutLabel label, int index, float orderKey) {
			this.box = box;
			this.score = score;
			this.label = label;
			this.index = index;
			this.orderKey = orderKey;
		}
	}
}
