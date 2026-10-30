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
package net.dreamlu.mica.ai.track.core;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.track.algo.HungarianMatcher;
import net.dreamlu.mica.ai.track.config.ByteTrackConfig;
import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * ByteTrack 多目标跟踪器（MIT 协议原论文复刻，零模型零 ONNX 依赖）。
 *
 * <p>每次调用 {@link #update(List)} 输入检测器吐的一帧 {@code List<BBox>}，
 * 输出当前帧的活跃轨迹（带稳定 ID）。内部状态逐帧推进。
 *
 * <p>核心步骤（与 ifzhang/ByteTrack 原版对齐）：
 * <ol>
 *   <li>所有已有轨迹 Kalman {@code predict()}</li>
 *   <li>第一轮匈牙利匹配：<b>高分 det ↔ 所有轨迹</b>（含 newborn）</li>
 *   <li>第二轮匈牙利匹配：<b>低分 det ↔ 第一轮没匹配上的轨迹</b>（成熟 + newborn）
 *       <br>这是 ByteTrack 灵魂——低分 det（运动模糊漏检 → score 掉下来的目标）
 *       不会被丢弃，再跟第一轮没匹配上的轨迹凑一次。</li>
 *   <li>两轮都没命中的 det → 新建轨迹（age=1）</li>
 *   <li>没被任何 det 匹配的轨迹 {@code onUnmatched()}；连续
 *       {@code trackBufferedThresh} 帧未匹配的轨迹 → 删除</li>
 * </ol>
 *
 * <p>线程安全：<b>不是</b>线程安全——内部用普通 {@link HashMap}，更新时逐帧重建。
 * 调用方做外部同步（跟踪器通常单视频流单线程跑）。
 *
 * <p>License：MIT（ifzhang/ByteTrack 原仓库）。
 */
@Slf4j
public class ByteTracker {

	private final ByteTrackConfig config;
	private final AtomicInteger nextId = new AtomicInteger(1);

	/**
	 * 当前活跃轨迹 Map（id → Track）。普通 HashMap，每帧 rebuild，非线程安全。
	 */
	private Map<Integer, Track> tracks = new HashMap<>();

	public ByteTracker(ByteTrackConfig config) {
		if (config == null) {
			throw new IllegalArgumentException("ByteTrackConfig must not be null");
		}
		config.validate();
		this.config = config;
		log.info("ByteTracker 初始化完成: high={}, low={}, match={}, matchLow={}, confirmAge={}, buffered={}",
			config.getHighThreshold(), config.getLowThreshold(), config.getMatchThresh(),
			config.getMatchLowThresh(), config.getConfirmAge(), config.getTrackBufferedThresh());
	}

	public static ByteTracker create(ByteTrackConfig config) {
		return new ByteTracker(config);
	}

	public List<Track> update(List<BBox> detections) {
		// 0. 保存旧轨迹快照（后面第 6 步 onUnmatched 只作用于旧轨迹，新轨迹 age=1 lostFrames=0 不需要）
		Map<Integer, Track> oldTracks = new HashMap<>(tracks);

		// 1. 所有轨迹 predict
		List<Track> allTrks = new ArrayList<>(oldTracks.values());
		for (Track t : allTrks) {
			t.getKf().predict();
		}

		// 2. dets 按 score 分高分 / 低分（lowThreshold 以下的连第二轮都进不去）
		List<BBox> highDets = new ArrayList<>();
		List<BBox> lowDets = new ArrayList<>();
		if (detections != null) {
			for (BBox d : detections) {
				if (d.getScore() >= config.getHighThreshold()) {
					highDets.add(d);
				} else if (d.getScore() >= config.getLowThreshold()) {
					lowDets.add(d);
				}
			}
		}

		Set<Integer> matchedThisFrame = new HashSet<>();

		// 3. 第一轮：高分 det ↔ 所有轨迹（mature + newborn 都可以被匹配并 age+=1）
		List<BBox> unmatchedHigh = matchAndMarkMatched(highDets, allTrks,
			config.getMatchThresh(), matchedThisFrame);

		// 4. 第二轮：低分 det ↔ 第一轮没匹配上的轨迹
		List<Track> unmatchedTrks = new ArrayList<>();
		for (Track t : allTrks) {
			if (!matchedThisFrame.contains(t.getId())) {
				unmatchedTrks.add(t);
			}
		}
		List<BBox> unmatchedLow = matchAndMarkMatched(lowDets, unmatchedTrks,
			config.getMatchLowThresh(), matchedThisFrame);

		// 5. 两轮都没命中的 det → 新建轨迹（age=1，lostFrames=0）
		for (BBox d : unmatchedHigh) {
			int id = nextId.getAndIncrement();
			Track t = new Track(id, d, 1, 0);
			allTrks.add(t);
		}
		for (BBox d : unmatchedLow) {
			int id = nextId.getAndIncrement();
			Track t = new Track(id, d, 1, 0);
			allTrks.add(t);
		}

		// 6. 本轮没被匹配的旧轨迹（新建的 age=1 lostFrames=0 不用 onUnmatched）：
		//    已匹配的 -> onMatched 已在 matchAndMarkMatched 里调过；
		//    未匹配的 -> onUnmatched() + 超限 markFinished
		for (Track t : oldTracks.values()) {
			if (!matchedThisFrame.contains(t.getId())) {
				t.onUnmatched();
				if (t.getLostFrames() > config.getTrackBufferedThresh()) {
					t.markFinished();
				}
			}
		}

		// 7. 移除 finished 轨迹（rebuilt 包含旧+新）
		Map<Integer, Track> rebuilt = new HashMap<>();
		for (Track t : allTrks) {
			if (!t.isFinished()) {
				rebuilt.put(t.getId(), t);
			}
		}
		this.tracks = rebuilt;

		// 8. 返回活跃轨迹（成熟优先、ID 递增）
		List<Track> active = new ArrayList<>(tracks.values());
		active.sort((a, b) -> {
			boolean ma = a.isMature(config.getConfirmAge());
			boolean mb = b.isMature(config.getConfirmAge());
			if (ma != mb) {
				return ma ? -1 : 1;
			}
			return Integer.compare(a.getId(), b.getId());
		});
		return active;
	}

	/**
	 * 匈牙利匹配 det ↔ trk，匹配成功的 trk 调 {@link Track#onMatched(BBox)}
	 * 并登记 id 到 {@code matchedThisFrame}。
	 *
	 * <p>返回值：<b>没被匹配的 det</b>（留给调用方看要不要新建轨迹）。
	 */
	private List<BBox> matchAndMarkMatched(List<BBox> dets, List<Track> trks, double iouThreshold,
										  Set<Integer> matchedThisFrame) {
		if (dets.isEmpty()) {
			// dets 空 → 本轮 trks 一个都没匹配上，不做任何标记（等调用方统一 onUnmatched）
			return new ArrayList<>();
		}
		if (trks.isEmpty()) {
			return new ArrayList<>(dets);
		}

		int m = dets.size();
		int n = trks.size();
		double[][] cost = new double[m][n];
		for (int i = 0; i < m; i++) {
			BBox d = dets.get(i);
			for (int j = 0; j < n; j++) {
				BBox pred = trks.get(j).getKf().predictedBBox();
				float iou = d.iou(pred);
				cost[i][j] = 1.0 - iou;
			}
		}

		// HungarianMatcher 内部 threshold 语义：cost<=threshold 才参与匹配
		// 调用方传 iouThreshold（0~1），转换成 cost 阈值 = 1 - iouThreshold
		HungarianMatcher.Result r = HungarianMatcher.match(cost, 1.0 - iouThreshold);

		for (int[] pair : r.matched) {
			BBox d = dets.get(pair[0]);
			Track t = trks.get(pair[1]);
			t.onMatched(d);
			matchedThisFrame.add(t.getId());
		}

		List<BBox> unmatched = new ArrayList<>();
		for (int detIdx : r.unmatchedDets) {
			unmatched.add(dets.get(detIdx));
		}
		return unmatched;
	}

	public ByteTrackConfig getConfig() {
		return config;
	}

	public int trackCount() {
		return tracks.size();
	}

	public void reset() {
		tracks = new HashMap<>();
		nextId.set(1);
		log.info("ByteTracker reset");
	}
}
