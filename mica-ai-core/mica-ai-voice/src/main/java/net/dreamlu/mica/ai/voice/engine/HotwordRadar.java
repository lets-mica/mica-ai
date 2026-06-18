package net.dreamlu.mica.ai.voice.engine;

import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.regex.Pattern;

/**
 * Trie 树加速版高性能热词召回组件。
 *
 * <p>对应 Python 端 {@code HotwordRadar}：
 * <ul>
 *   <li>字符级 Trie 树合并所有热词前缀</li>
 *   <li>基于 Trie 节点的子节点字典快速剪枝</li>
 *   <li>多维优先级覆盖去重</li>
 * </ul>
 */
@Slf4j
public final class HotwordRadar {

	private static final Pattern NON_WORD = Pattern.compile("[^\\w\\s]+");
	private static final Pattern WHITESPACE = Pattern.compile("\\s+");
	private static final double FRAME_DURATION = 0.060;

	private final SentencePieceTokenizer tokenizer;

	/** 预计算全量词表的小写映射 */
	private final String[] vocabLower;

	/** 当前热词列表 */
	private List<String> hotwords;
	/** 用于搜索的热词（去除非文字字符） */
	private String[] searchHotwords;
	/** 小写无空格的热词字符串 */
	private String[] hotwordLowerStrings;
	/** Trie 根节点 */
	private TrieNode trie;

	public HotwordRadar(List<String> hotwords, SentencePieceTokenizer tokenizer) {
		this.tokenizer = tokenizer;

		// 预计算词表小写映射
		int vocabSize = tokenizer.getPieceSize();
		this.vocabLower = new String[vocabSize];
		for (int i = 0; i < vocabSize; i++) {
			String piece = tokenizer.idToPiece(i);
			vocabLower[i] = piece.toLowerCase().replace('\u2581', ' ').trim();
		}

		updateHotwords(hotwords != null ? hotwords : List.of());
	}

	/**
	 * 动态更新热词列表并重构 Trie 树。
	 */
	public void updateHotwords(List<String> hotwords) {
		this.hotwords = new ArrayList<>(hotwords);
		this.trie = new TrieNode();
		this.searchHotwords = new String[hotwords.size()];
		this.hotwordLowerStrings = new String[hotwords.size()];

		for (int idx = 0; idx < hotwords.size(); idx++) {
			String w = hotwords.get(idx);
			searchHotwords[idx] = NON_WORD.matcher(w).replaceAll(" ");
			String clean = WHITESPACE.matcher(searchHotwords[idx]).replaceAll("").toLowerCase();
			hotwordLowerStrings[idx] = clean;
			if (clean.isEmpty()) continue;

			// 插入 Trie
			TrieNode node = trie;
			for (int c = 0; c < clean.length(); c++) {
				char ch = clean.charAt(c);
				node = node.children.computeIfAbsent(ch, k -> new TrieNode());
			}
			node.wordIndices.add(idx);
		}
	}

	/**
	 * 扫描 Top-K 解码输出，检测热词匹配。
	 *
	 * @param radarIndices Top-K token 索引 [T][K]
	 * @param radarProbs   Top-K 概率 [T][K]
	 * @param topK         实际搜索深度
	 * @return 检测到的热词列表
	 */
	public List<HotwordHit> scan(int[][] radarIndices, float[][] radarProbs, int topK) {
		int t = radarIndices.length;
		if (t == 0) return List.of();

		int k = Math.min(topK, radarIndices[0].length);

		// 提取 Top-1（Greedy 非空帧判断基准）
		int[] top1 = new int[t];
		for (int i = 0; i < t; i++) {
			top1[i] = radarIndices[i][0];
		}

		List<RawHit> hits = new ArrayList<>();

		for (int frame = 0; frame < t; frame++) {
			if (top1[frame] == 0) continue; // blank

			Set<String> seenTokens = new HashSet<>();
			for (int ki = 0; ki < k; ki++) {
				int tid = radarIndices[frame][ki];
				String tc = vocabLower[tid];
				if (tc.isEmpty() || seenTokens.contains(tc)) continue;
				seenTokens.add(tc);

				// 检查首字符是否在 Trie 根部
				if (tc.length() > 0 && trie.children.containsKey(tc.charAt(0))) {
					// 获取原始 piece 的词边界标记
					String originalPiece = tokenizer.idToPiece(tid);
					boolean hasBoundary = originalPiece.startsWith("\u2581");

					// 尝试在 Trie 上匹配此 token 的全部字符
					TrieNode node = trie;
					boolean matchPossible = true;
					for (int c = 0; c < tc.length(); c++) {
						TrieNode child = node.children.get(tc.charAt(c));
						if (child != null) {
							node = child;
						} else {
							matchPossible = false;
							break;
						}
					}

					if (matchPossible) {
						// DFS 继续往后搜索
						Map<String, Object> memo = new HashMap<>();
						List<RawHit> frameHits = dfsTrie(frame, ki, node, radarIndices, radarProbs,
							top1, k, t, memo);
						for (RawHit h : frameHits) {
							h.hasWordBoundary = hasBoundary;
							hits.add(h);
						}
					}
				}
			}
		}

		return postProcess(hits, top1);
	}

	// ==================== DFS Trie 搜索 ====================

	private List<RawHit> dfsTrie(int tCurr, int kCurr, TrieNode startNode,
								 int[][] topkIds, float[][] topkProbs,
								 int[] top1, int k, int totalT,
								 Map<String, Object> memo) {
		float pStart = topkProbs[tCurr][kCurr];
		String t1 = vocabLower[topkIds[tCurr][kCurr]];
		int maxLookahead = 15;

		List<RawHit> results = new ArrayList<>();

		// 递归搜索函数（用迭代式 DFS 模拟）
		searchFromFrame(tCurr, startNode, topkIds, topkProbs, top1, k, totalT,
			maxLookahead, memo, t1, pStart, tCurr, results);

		return results;
	}

	@SuppressWarnings("unchecked")
	private void searchFromFrame(int tCurr, TrieNode startNode,
								 int[][] topkIds, float[][] topkProbs,
								 int[] top1, int k, int totalT,
								 int maxLookahead, Map<String, Object> memo,
								 String firstToken, float pStart, int startFrame,
								 List<RawHit> results) {
		// 使用递归 DFS
		dfsRecurse(startFrame, startNode, topkIds, topkProbs, top1, k, totalT,
			maxLookahead, memo, firstToken, pStart, startFrame, results,
			new ArrayList<>(), new ArrayList<>(), 0);
	}

	@SuppressWarnings("unchecked")
	private void dfsRecurse(int fPrev, TrieNode node,
							int[][] topkIds, float[][] topkProbs,
							int[] top1, int k, int totalT,
							int maxLookahead, Map<String, Object> memo,
							String firstToken, float pStart, int startFrame,
							List<RawHit> results,
							List<Integer> frameIndices, List<String> matchedTokens,
							float probSum) {
		// 检查当前节点是否为热词终点
		for (int wIdx : node.wordIndices) {
			RawHit hit = new RawHit();
			hit.wordIdx = wIdx;
			hit.startFrame = startFrame;
			hit.endFrame = fPrev;
			hit.prob = (probSum + pStart) / (frameIndices.size() + 1);
			hit.frameIndices = new ArrayList<>();
			hit.frameIndices.add(startFrame);
			hit.frameIndices.addAll(frameIndices);
			hit.matchedTokens = new ArrayList<>();
			hit.matchedTokens.add(firstToken);
			hit.matchedTokens.addAll(matchedTokens);
			results.add(hit);
		}

		// 继续往后搜索
		int searchEnd = Math.min(fPrev + 1 + maxLookahead, totalT);
		for (int f = fPrev + 1; f < searchEnd; f++) {
			// 如果中间有非 blank 的 Greedy 帧，停止（CTC 对齐约束）
			if (f > fPrev + 1) {
				boolean hasNonBlank = false;
				for (int ff = fPrev + 1; ff < f; ff++) {
					if (top1[ff] != 0) {
						hasNonBlank = true;
						break;
					}
				}
				if (hasNonBlank) break;
			}

			for (int ki = 0; ki < k; ki++) {
				String tc = vocabLower[topkIds[f][ki]];
				if (tc.isEmpty()) continue;

				// 在 Trie 上匹配
				TrieNode tempNode = node;
				boolean matchOk = true;
				for (int c = 0; c < tc.length(); c++) {
					TrieNode child = tempNode.children.get(tc.charAt(c));
					if (child != null) {
						tempNode = child;
					} else {
						matchOk = false;
						break;
					}
				}

				if (matchOk) {
					frameIndices.add(f);
					matchedTokens.add(tc);
					float newProbSum = probSum + topkProbs[f][ki];

					dfsRecurse(f, tempNode, topkIds, topkProbs, top1, k, totalT,
						maxLookahead, memo, firstToken, pStart, startFrame, results,
						frameIndices, matchedTokens, newProbSum);

					// 回溯
					frameIndices.remove(frameIndices.size() - 1);
					matchedTokens.remove(matchedTokens.size() - 1);
				}
			}
		}
	}

	// ==================== 后处理 ====================

	private List<HotwordHit> postProcess(List<RawHit> hits, int[] top1) {
		if (hits.isEmpty()) return List.of();

		// 1. 基础过滤
		List<RawHit> filtered = new ArrayList<>();
		for (RawHit h : hits) {
			int nbGreedy = 0;
			for (int f : h.frameIndices) {
				if (top1[f] != 0) nbGreedy++;
			}
			int bGreedy = h.frameIndices.size() - nbGreedy;

			if (nbGreedy >= 2) {
				h.nbGreedy = nbGreedy;
				h.bGreedy = bGreedy;
				filtered.add(h);
			}
		}
		if (filtered.isEmpty()) return List.of();

		// 2. 排序与多维优先级覆盖去重
		filtered.sort(Comparator.comparingInt(h -> h.startFrame));
		List<RawHit> selected = new ArrayList<>();
		int i = 0;
		while (i < filtered.size()) {
			RawHit best = filtered.get(i);
			int j = i + 1;
			while (j < filtered.size() && filtered.get(j).startFrame <= best.endFrame) {
				RawHit candidate = filtered.get(j);

				if (candidate.nbGreedy > best.nbGreedy) {
					best = candidate;
				} else if (candidate.nbGreedy == best.nbGreedy) {
					if (candidate.bGreedy < best.bGreedy) {
						best = candidate;
					} else if (candidate.bGreedy == best.bGreedy) {
						int lenC = hotwordLowerStrings[candidate.wordIdx].length();
						int lenB = hotwordLowerStrings[best.wordIdx].length();
						if (lenC > lenB) {
							best = candidate;
						} else if (lenC == lenB && candidate.prob > best.prob) {
							best = candidate;
						}
					}
				}
				j++;
			}
			selected.add(best);
			i = j;
		}

		// 3. 格式化输出
		List<HotwordHit> final_ = new ArrayList<>();
		for (RawHit h : selected) {
			String text = hotwords.get(h.wordIdx);
			if (h.hasWordBoundary) {
				text = " " + text;
			}

			List<TokenTime> tokens = new ArrayList<>();
			for (int idx = 0; idx < h.matchedTokens.size(); idx++) {
				tokens.add(new TokenTime(
					h.matchedTokens.get(idx),
					Math.round(h.frameIndices.get(idx) * FRAME_DURATION * 1000.0) / 1000.0
				));
			}

			final_.add(new HotwordHit(
				text,
				Math.round(h.startFrame * FRAME_DURATION * 1000.0) / 1000.0,
				Math.round(h.endFrame * FRAME_DURATION * 1000.0) / 1000.0,
				Math.round(h.prob * 10000.0) / 10000.0,
				tokens
			));
		}
		return final_;
	}

	// ==================== 内部数据结构 ====================

	static final class TrieNode {
		final Map<Character, TrieNode> children = new HashMap<>();
		final List<Integer> wordIndices = new ArrayList<>();
	}

	static final class RawHit {
		int wordIdx;
		int startFrame;
		int endFrame;
		double prob;
		List<Integer> frameIndices;
		List<String> matchedTokens;
		boolean hasWordBoundary;
		int nbGreedy;
		int bGreedy;
	}

	public record TokenTime(String token, double time) {
	}

	public record HotwordHit(String text, double start, double end, double prob,
							 List<TokenTime> tokens) {
	}
}

