package net.dreamlu.mica.ai.voice.engine;

import net.dreamlu.mica.ai.voice.config.RecognitionResult;

import java.util.*;

/**
 * 结果整合器：将 Greedy 识别流与热词匹配流进行无缝融合与替换。
 *
 * <p>对应 Python 端 {@code ResultIntegrator}：
 * <ol>
 *   <li>预处理：过滤掉与前一个热词时间重叠的热词（保留最早出现的）</li>
 *   <li>双指针合并：遍历 greedy token，热词指针只向前推进，O(N+M)</li>
 * </ol>
 */
public final class ResultIntegrator {

	/**
	 * 整合 Greedy 结果与热词检测结果。
	 *
	 * @param greedyResults   Greedy 解码结果列表
	 * @param detectedHotwords 热词雷达检测到的热词
	 * @return 整合后的识别结果列表
	 */
	public static List<RecognitionResult> integrate(
		List<SenseVoiceDecoder.GreedyItem> greedyResults,
		List<HotwordRadar.HotwordHit> detectedHotwords) {

		// 步骤 1：过滤重叠热词，保留时间最早的那个
		List<HotwordRadar.HotwordHit> sortedHotwords = new ArrayList<>(detectedHotwords);
		sortedHotwords.sort(Comparator.comparingDouble(HotwordRadar.HotwordHit::start));

		List<HotwordRadar.HotwordHit> activeHotwords = new ArrayList<>();
		double lastEnd = -1.0;
		for (HotwordRadar.HotwordHit hw : sortedHotwords) {
			if (hw.start() >= lastEnd - 0.02) {
				activeHotwords.add(hw);
				lastEnd = hw.end();
			}
		}

		// 步骤 2：双指针合并
		List<RecognitionResult> finalResults = new ArrayList<>();
		int hwIdx = 0;
		Set<Integer> emitted = new HashSet<>();

		for (SenseVoiceDecoder.GreedyItem g : greedyResults) {
			double gStart = g.start();

			// 推进热词指针：跳过已完全结束的热词
			while (hwIdx < activeHotwords.size()
				&& activeHotwords.get(hwIdx).end() + 0.02 < gStart) {
				hwIdx++;
			}

			// 判断当前 greedy token 是否落在热词区间内
			boolean inHotwordSpan = false;
			if (hwIdx < activeHotwords.size()) {
				HotwordRadar.HotwordHit hw = activeHotwords.get(hwIdx);
				if (hw.start() - 0.02 <= gStart && gStart <= hw.end() + 0.02) {
					inHotwordSpan = true;
					if (!emitted.contains(hwIdx)) {
						// 首次进入该热词区间：输出热词块
						finalResults.addAll(mergeTokensToChunks(hw));
						emitted.add(hwIdx);
					}
				}
			}

			if (!inHotwordSpan) {
				finalResults.add(new RecognitionResult(g.text(), g.start(), false));
			}
		}

		return finalResults;
	}

	/**
	 * 将热词内部的 Token 和原始文本进行"块对齐"切分。
	 */
	private static List<RecognitionResult> mergeTokensToChunks(HotwordRadar.HotwordHit hw) {
		String originText = hw.text();
		String searchBase = originText.toLowerCase();
		List<RecognitionResult> chunks = new ArrayList<>();

		// 1. 寻找每个 Token 覆盖的字符起始位置
		List<int[]> anchors = new ArrayList<>(); // [idx_in_text, timestamp_idx]
		List<Double> anchorTimes = new ArrayList<>();
		int currSearchPos = 0;

		for (HotwordRadar.TokenTime tk : hw.tokens()) {
			String cleanTk = tk.token().replace('\u2581', ' ').trim().toLowerCase();
			if (cleanTk.isEmpty()) continue;
			int idx = searchBase.indexOf(cleanTk, currSearchPos);
			if (idx != -1) {
				anchors.add(new int[]{idx});
				anchorTimes.add(tk.time());
				currSearchPos = idx + cleanTk.length();
			}
		}

		if (anchors.isEmpty()) {
			anchors.add(new int[]{0});
			anchorTimes.add(hw.start());
		} else if (anchors.get(0)[0] != 0) {
			anchors.add(0, new int[]{0});
			anchorTimes.add(0, hw.start());
		}

		// 2. 根据锚点切割原始文本块
		for (int i = 0; i < anchors.size(); i++) {
			int startIdx = anchors.get(i)[0];
			double startTime = anchorTimes.get(i);
			int nextIdx = (i + 1 < anchors.size()) ? anchors.get(i + 1)[0] : originText.length();

			String chunkText = originText.substring(startIdx, nextIdx);
			chunks.add(new RecognitionResult(chunkText, startTime, true));
		}
		return chunks;
	}
}

