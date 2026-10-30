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

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link HungarianMatcher} 贪心匈牙利单测。
 *
 * <p>成本矩阵语义：{@code cost[i][j] = 1 - IoU(det_i, trk_j)}，越小越好；
 * 超过阈值的条目先置 INF。
 */
class HungarianMatcherTest {

	@Test
	void emptyMatrixReturnsNoMatches() {
		HungarianMatcher.Result r = HungarianMatcher.match(new double[0][0], 0.8);
		assertThat(r.matched.length).isEqualTo(0);
		assertThat(r.unmatchedDets.length).isEqualTo(0);
		assertThat(r.unmatchedTrks.length).isEqualTo(0);
	}

	@Test
	void oneToOnePerfectMatch() {
		double[][] cost = {
			{0.0, 1.0},
			{1.0, 0.0}
		};
		HungarianMatcher.Result r = HungarianMatcher.match(cost, 0.5);
		assertThat(r.matched.length).isEqualTo(2);
		assertThat(r.unmatchedDets.length).isEqualTo(0);
		assertThat(r.unmatchedTrks.length).isEqualTo(0);
	}

	@Test
	void thresholdRejectsBadPairs() {
		// cost=0.3 应被拒绝 → HungarianMatcher 传 threshold=0.2（只接受 cost<=0.2）
		double[][] cost = {{0.3}};
		HungarianMatcher.Result r = HungarianMatcher.match(cost, 0.2);
		assertThat(r.matched.length).isEqualTo(0);
		assertThat(r.unmatchedDets.length).isEqualTo(1);
		assertThat(r.unmatchedDets[0]).isEqualTo(0);
		assertThat(r.unmatchedTrks.length).isEqualTo(1);
		assertThat(r.unmatchedTrks[0]).isEqualTo(0);
	}

	@Test
	void thresholdAcceptsGoodPairs() {
		double[][] cost = {{0.1}};
		HungarianMatcher.Result r = HungarianMatcher.match(cost, 0.8);
		assertThat(r.matched.length).isEqualTo(1);
		assertThat(r.unmatchedDets.length).isEqualTo(0);
		assertThat(r.unmatchedTrks.length).isEqualTo(0);
	}

	@Test
	void moreDetsThanTrks() {
		double[][] cost = {
			{0.1, 0.9},
			{0.9, 0.1},
			{0.5, 0.5}
		};
		HungarianMatcher.Result r = HungarianMatcher.match(cost, 0.8);
		assertThat(r.matched.length).isEqualTo(2);
		assertThat(r.unmatchedDets.length).isEqualTo(1);
		assertThat(r.unmatchedDets[0]).isEqualTo(2);
		assertThat(r.unmatchedTrks.length).isEqualTo(0);
	}

	@Test
	void moreTrksThanDets() {
		double[][] cost = {
			{0.1, 0.9, 0.8}
		};
		HungarianMatcher.Result r = HungarianMatcher.match(cost, 0.8);
		assertThat(r.matched.length).isEqualTo(1);
		assertThat(r.unmatchedDets.length).isEqualTo(0);
		assertThat(r.unmatchedTrks.length).isEqualTo(2);
		boolean has1 = false, has2 = false;
		for (int t : r.unmatchedTrks) {
			if (t == 1) has1 = true;
			if (t == 2) has2 = true;
		}
		assertThat(has1 && has2).isTrue();
	}

	@Test
	void zeroThresholdForcesNothing() {
		double[][] cost = {{0.1}};
		HungarianMatcher.Result r = HungarianMatcher.match(cost, 0.0);
		assertThat(r.matched.length).isEqualTo(0);
	}

	@Test
	void oneDetNoTrks() {
		double[][] cost = {{}};
		HungarianMatcher.Result r = HungarianMatcher.match(cost, 0.8);
		assertThat(r.matched.length).isEqualTo(0);
		assertThat(r.unmatchedDets.length).isEqualTo(1);
		assertThat(r.unmatchedDets[0]).isEqualTo(0);
		assertThat(r.unmatchedTrks.length).isEqualTo(0);
	}

	@Test
	void allRowsPickDistinctCols() {
		double[][] cost = new double[5][3];
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 3; j++) {
				cost[i][j] = (i == j) ? 0.1 : 0.9;
			}
		}
		HungarianMatcher.Result r = HungarianMatcher.match(cost, 0.8);
		assertThat(r.matched.length).isEqualTo(3);
		assertThat(r.unmatchedDets.length).isEqualTo(2);
		boolean has3 = false, has4 = false;
		for (int d : r.unmatchedDets) {
			if (d == 3) has3 = true;
			if (d == 4) has4 = true;
		}
		assertThat(has3 && has4).isTrue();
	}
}
