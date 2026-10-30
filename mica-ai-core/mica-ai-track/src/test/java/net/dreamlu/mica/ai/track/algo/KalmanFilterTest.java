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

import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link KalmanFilter} + {@link Track} 基础逻辑单测。
 *
 * <p>不追求数值精确到小数点后多少位（Kalman 是近似解），只验证语义正确：
 * predict 后有合理位置；update 后位置会跟观测靠近；连续多帧静止观测下最终收敛。
 */
class KalmanFilterTest {

	@Test
	void predictStaysNearInitialWithoutMotion() {
		BBox init = new BBox(100f, 100f, 150f, 180f, 0.9f);
		KalmanFilter kf = new KalmanFilter(init);
		for (int i = 0; i < 5; i++) {
			kf.predict();
		}
		// 没给速度观测，vx/vy/vw/vh 应保持 ~0，预测位置不应漂移到离谱范围
		assertThat(kf.predictedCx()).isBetween(120d, 130d);
		assertThat(kf.predictedCy()).isBetween(135d, 145d);
	}

	@Test
	void updatePullsPredictionTowardsObservation() {
		BBox init = new BBox(0f, 0f, 50f, 50f, 0.9f);
		KalmanFilter kf = new KalmanFilter(init);
		BBox farObs = new BBox(200f, 200f, 250f, 250f, 0.9f);

		// predict 一次（还没动）→ 位置仍是 ~25,25
		kf.predict();
		double beforeCx = kf.predictedCx();
		double beforeCy = kf.predictedCy();
		assertThat(beforeCx).isBetween(20d, 30d);

		// 用远观测 update → 预测 cx 应该朝 225 靠近（卡尔曼加权，不会直接跳过去）
		kf.update(farObs);
		assertThat(kf.predictedCx()).isGreaterThan(beforeCx);
		assertThat(kf.predictedCy()).isGreaterThan(beforeCy);
	}

	@Test
	void stationaryObservationsConverge() {
		BBox init = new BBox(0f, 0f, 50f, 50f, 0.9f);
		BBox obs = new BBox(100f, 100f, 150f, 150f, 0.9f);
		KalmanFilter kf = new KalmanFilter(init);
		for (int i = 0; i < 20; i++) {
			kf.predict();
			kf.update(obs);
		}
		// 卡尔曼应收敛到观测附近（cx ≈ 125）
		assertThat(kf.predictedCx()).isBetween(115d, 135d);
		assertThat(kf.predictedCy()).isBetween(115d, 135d);
	}

	@Test
	void bboxFromStateIsPositiveSized() {
		KalmanFilter kf = new KalmanFilter(new BBox(100f, 100f, 200f, 200f, 0.9f));
		for (int i = 0; i < 100; i++) {
			BBox b = kf.predictedBBox();
			assertThat(b.isEmpty()).isFalse();
			assertThat(b.width()).isGreaterThan(0f);
			assertThat(b.height()).isGreaterThan(0f);
		}
	}

	// --- Track 基础 ---

	@Test
	void trackNewbornIsNotMature() {
		Track t = new Track(1, new BBox(0f, 0f, 50f, 50f, 0.9f), 1, 0);
		assertThat(t.isMature(3)).isFalse();
	}

	@Test
	void trackBecomesMatureAfterEnoughMatches() {
		Track t = new Track(1, new BBox(0f, 0f, 50f, 50f, 0.9f), 1, 0);
		t.onMatched(new BBox(1f, 1f, 51f, 51f, 0.9f));
		t.onMatched(new BBox(2f, 2f, 52f, 52f, 0.9f));
		// age = 3 正好等于 confirmAge 3
		assertThat(t.isMature(3)).isTrue();
	}

	@Test
	void trackUnmatchedIncrementsLostFrames() {
		Track t = new Track(1, new BBox(0f, 0f, 50f, 50f, 0.9f), 5, 0);
		t.onUnmatched();
		t.onUnmatched();
		assertThat(t.getLostFrames()).isEqualTo(2);
	}

	@Test
	void onMatchedResetsLostFrames() {
		Track t = new Track(1, new BBox(0f, 0f, 50f, 50f, 0.9f), 5, 3);
		t.onMatched(new BBox(1f, 1f, 51f, 51f, 0.9f));
		assertThat(t.getLostFrames()).isEqualTo(0);
	}
}
