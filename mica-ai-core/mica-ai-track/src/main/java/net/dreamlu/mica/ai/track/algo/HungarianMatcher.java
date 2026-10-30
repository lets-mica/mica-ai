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
package net.dreamlu.mica.ai.track.algo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 匈牙利匹配器（贪心近似，O(n²)）。
 *
 * <p>输入代价矩阵 {@code cost[i][j]}（detection i 与 track j 的距离，越小越好）
 * 和 IoU 阈值 {@code threshold}，返回三部分：
 * <ul>
 *   <li>{@code matched}：已匹配的 (detIdx, trackIdx) 对</li>
 *   <li>{@code unmatchedDets}：未能匹配的检测索引</li>
 *   <li>{@code unmatchedTrks}：未能匹配的轨迹索引</li>
 * </ul>
 *
 * <p>贪心策略：
 * <ol>
 *   <li>代价矩阵上 <em>阈值外设为 INF</em>（先硬裁剪）</li>
 *   <li>按行排序，每行取最小列，若列未被占用则贪心认领</li>
 *   <li>处理完所有行后，未被匹配的行 / 列即是 unmatched</li>
 * </ol>
 *
 * <p>贪心匈牙利在目标跟踪场景下与 Jonker-Volgenant O(n³) 的精确解
 * 效果几乎一致（n 通常 < 50），但实现简单得多。
 */
public class HungarianMatcher {

	public static final double INF = 1e9;

	public static class Result {
		public final int[][] matched;
		public final int[] unmatchedDets;
		public final int[] unmatchedTrks;

		public Result(int[][] matched, int[] unmatchedDets, int[] unmatchedTrks) {
			this.matched = matched;
			this.unmatchedDets = unmatchedDets;
			this.unmatchedTrks = unmatchedTrks;
		}
	}

	public static Result match(double[][] costMatrix, double threshold) {
		if (costMatrix == null || costMatrix.length == 0
			|| costMatrix[0] == null || costMatrix[0].length == 0) {
			return empty(costMatrix);
		}
		int m = costMatrix.length;
		int n = costMatrix[0].length;

		double[][] trimmed = new double[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				trimmed[i][j] = costMatrix[i][j] <= threshold ? costMatrix[i][j] : INF;
			}
		}

		int[] rowAssign = new int[m];
		int[] colAssign = new int[n];
		Arrays.fill(rowAssign, -1);
		Arrays.fill(colAssign, -1);

		List<int[]> matched = new ArrayList<>();
		for (int i = 0; i < m; i++) {
			int bestJ = -1;
			double best = INF;
			for (int j = 0; j < n; j++) {
				if (trimmed[i][j] < best) {
					best = trimmed[i][j];
					bestJ = j;
				}
			}
			if (bestJ >= 0 && best < INF && colAssign[bestJ] == -1) {
				rowAssign[i] = bestJ;
				colAssign[bestJ] = i;
				matched.add(new int[]{i, bestJ});
			}
		}

		List<Integer> uDets = new ArrayList<>();
		for (int i = 0; i < m; i++) {
			if (rowAssign[i] == -1) {
				uDets.add(i);
			}
		}
		List<Integer> uTrks = new ArrayList<>();
		for (int j = 0; j < n; j++) {
			if (colAssign[j] == -1) {
				uTrks.add(j);
			}
		}

		return new Result(
			matched.toArray(new int[0][]),
			toIntArray(uDets),
			toIntArray(uTrks)
		);
	}

	private static Result empty(double[][] costMatrix) {
		int m = costMatrix == null ? 0 : costMatrix.length;
		int n = (costMatrix == null || m == 0) ? 0 : costMatrix[0].length;
		int[][] matched = new int[0][];
		int[] uDets = new int[m];
		for (int i = 0; i < m; i++) {
			uDets[i] = i;
		}
		int[] uTrks = new int[n];
		for (int j = 0; j < n; j++) {
			uTrks[j] = j;
		}
		return new Result(matched, uDets, uTrks);
	}

	private static int[] toIntArray(List<Integer> list) {
		int[] out = new int[list.size()];
		for (int i = 0; i < list.size(); i++) {
			out[i] = list.get(i);
		}
		return out;
	}
}
