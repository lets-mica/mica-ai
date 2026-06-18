/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent.tokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 * BERT 中文分词器，按字切分。
 *
 * <p>分词策略：
 * <ul>
 *   <li>CJK 字符（包括中文汉字、日文假名、韩文）→ 逐字拆分</li>
 *   <li>英文字母和数字 → 连续作为整体</li>
 *   <li>其他字符（标点、空格等）→ 逐字符处理</li>
 * </ul>
 *
 * <p>处理流程：
 * <ol>
 *   <li>文本 → 字符粒度 token 列表</li>
 *   <li>各 token 查 vocab 获取 id，未知→[UNK]=100</li>
 *   <li>添加 [CLS]=101 和 [SEP]=102</li>
 *   <li>截断或 padding 到 maxLength</li>
 *   <li>构造 attention_mask 和 token_type_ids</li>
 * </ol>
 */
public final class BertTokenizer {

	private final VocabLoader vocab;
	private final int maxLength;

	/**
	 * 构造分词器。
	 *
	 * @param vocab     词表
	 * @param maxLength 最大序列长度（含 [CLS] 和 [SEP]）
	 */
	public BertTokenizer(VocabLoader vocab, int maxLength) {
		this.vocab = vocab;
		this.maxLength = maxLength;
	}

	/**
	 * 对文本进行分词并返回模型输入。
	 *
	 * @param text 输入文本
	 * @return TokenResult 包含 inputIds、attentionMask、tokenTypeIds
	 */
	public TokenResult tokenize(String text) {
		if (text == null || text.isEmpty()) {
			return createEmptyResult();
		}

		List<String> tokens = splitTokens(text);
		return encodeTokens(tokens);
	}

	/**
	 * 按 BERT 中文分词粒度拆分 token。
	 */
	private List<String> splitTokens(String text) {
		List<String> tokens = new ArrayList<>();
		int len = text.length();
		int i = 0;

		while (i < len) {
			char c = text.charAt(i);
			int codePoint = text.codePointAt(i);
			int charWidth = Character.charCount(codePoint);

			if (isCjk(codePoint)) {
				// CJK 字符：逐字拆分
				tokens.add(new String(Character.toChars(codePoint)));
		} else if (isAsciiLetterOrDigit(c)) {
			// 英文字母/数字（仅 ASCII）：连续作为整体
			int start = i;
			while (i < len) {
				char ch = text.charAt(i);
				if (isAsciiLetterOrDigit(ch)) {
					i++;
				} else {
					break;
				}
			}
			tokens.add(text.substring(start, i));
			continue; // 跳过末尾的 i++，因为循环内已经移动了 i
			} else {
				// 其他字符（标点、空格等）：单独处理，跳过空白
				if (!Character.isWhitespace(c)) {
					tokens.add(String.valueOf(c));
				}
			}
			i += charWidth;
		}

		return tokens;
	}

	/**
	 * 将 token 列表编码为 BERT 模型输入。
	 */
	private TokenResult encodeTokens(List<String> tokens) {
		int maxInputLen = maxLength - 2; // 预留 [CLS] 和 [SEP]
		int actualLen = Math.min(tokens.size(), maxInputLen);

		long[] inputIds = new long[maxLength];
		long[] attentionMask = new long[maxLength];
		long[] tokenTypeIds = new long[maxLength];

		// [CLS]
		inputIds[0] = VocabLoader.CLS_ID;
		attentionMask[0] = 1;

		// 中间 token
		for (int i = 0; i < actualLen; i++) {
			int id = vocab.getId(tokens.get(i));
			inputIds[i + 1] = id;
			attentionMask[i + 1] = 1;
		}

		// [SEP]
		int sepPos = actualLen + 1;
		inputIds[sepPos] = VocabLoader.SEP_ID;
		attentionMask[sepPos] = 1;

		// 剩余位置自动为 [PAD]=0，attention_mask=0, token_type_ids=0（数组默认值即 0）

		return new TokenResult(inputIds, attentionMask, tokenTypeIds);
	}

	/**
	 * 空文本兜底：只有 [CLS] 和 [SEP]。
	 */
	private TokenResult createEmptyResult() {
		long[] inputIds = new long[maxLength];
		long[] attentionMask = new long[maxLength];
		long[] tokenTypeIds = new long[maxLength];

		inputIds[0] = VocabLoader.CLS_ID;
		attentionMask[0] = 1;
		inputIds[1] = VocabLoader.SEP_ID;
		attentionMask[1] = 1;

		return new TokenResult(inputIds, attentionMask, tokenTypeIds);
	}

	/**
	 * 判断是否为 CJK 字符（中日韩统一表意文字、假名、韩文）。
	 */
	private static boolean isCjk(int codePoint) {
		return Character.isIdeographic(codePoint)
			|| (codePoint >= 0x3040 && codePoint <= 0x309F)  // 日文平假名
			|| (codePoint >= 0x30A0 && codePoint <= 0x30FF)  // 日文片假名
			|| (codePoint >= 0xAC00 && codePoint <= 0xD7AF); // 韩文
	}

	/**
	 * 判断是否为 ASCII 范围内的字母或数字。
	 * <p>注意：不能直接用 {@link Character#isLetter(char)}，因为 CJK 字符
	 * （Unicode OTHER_LETTER 类别）也会被判定为字母，导致英文 token 吞掉后续中文。
	 */
	private static boolean isAsciiLetterOrDigit(char c) {
		return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
	}

	/**
	 * 分词结果，包含 BERT 模型所需的三个输入张量。
	 */
	public record TokenResult(
		/** token ID 序列 [maxLength] */
		long[] inputIds,
		/** 注意力掩码 [maxLength]，实 token=1，padding=0 */
		long[] attentionMask,
		/** token 类型 ID [maxLength]，单句任务全为 0 */
		long[] tokenTypeIds
	) {
	}
}
