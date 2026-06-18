/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent;

import net.dreamlu.mica.ai.intent.tokenizer.BertTokenizer;
import net.dreamlu.mica.ai.intent.tokenizer.VocabLoader;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link BertTokenizer} 单元测试。
 */
@DisplayName("BertTokenizer 中文分词器")
class BertTokenizerTest {

	private static final ClassLoader CL = Thread.currentThread().getContextClassLoader();

	private static VocabLoader vocab;

	@BeforeAll
	static void loadVocab() throws IOException {
		vocab = VocabLoader.loadFromClasspath(CL, "test-vocab.txt");
	}

	// ==================== [CLS]/[SEP] 测试 ====================

	@Nested
	@DisplayName("[CLS] 和 [SEP] 位置")
	class ClsSepPlacement {

		@Test
		@DisplayName("首位置为 [CLS]=101")
		void testClsAtFirstPosition() {
			BertTokenizer tokenizer = new BertTokenizer(vocab, 64);
			BertTokenizer.TokenResult result = tokenizer.tokenize("你好");

			assertEquals(VocabLoader.CLS_ID, result.inputIds()[0]);
			assertEquals(1, result.attentionMask()[0], "[CLS] 的 attention_mask 应为 1");
		}

		@Test
		@DisplayName("末位 token 之后为 [SEP]=102")
		void testSepAfterTokens() {
			BertTokenizer tokenizer = new BertTokenizer(vocab, 64);
			BertTokenizer.TokenResult result = tokenizer.tokenize("你好");

			// "你好" 有 2 个 CJK 字符：[CLS] + 你 + 好 + [SEP] = 4 tokens
			// [SEP] 在索引 3
			assertEquals(VocabLoader.SEP_ID, result.inputIds()[3],
				"[SEP] 应在 token 序列之后");
			assertEquals(1, result.attentionMask()[3], "[SEP] 的 attention_mask 应为 1");
			assertEquals(VocabLoader.PAD_ID, result.inputIds()[4],
				"[SEP] 之后应为 [PAD]");
		}
	}

	// ==================== CJK 切分测试 ====================

	@Nested
	@DisplayName("CJK 按字切分")
	class CjkSplitting {

		private final BertTokenizer tokenizer = new BertTokenizer(vocab, 128);

		@Test
		@DisplayName("「今天天气怎么样」→ 逐字")
		void testChineseSentenceSplit() {
			BertTokenizer.TokenResult result = tokenizer.tokenize("今天天气怎么样");

			long[] ids = result.inputIds();
			assertEquals(VocabLoader.CLS_ID, ids[0]);                    // [CLS]
			assertEquals(vocab.getId("今"), ids[1]);                     // 今
			assertEquals(vocab.getId("天"), ids[2]);                     // 天
			assertEquals(vocab.getId("天"), ids[3]);                     // 天
			assertEquals(vocab.getId("气"), ids[4]);                     // 气
			assertEquals(vocab.getId("怎"), ids[5]);                     // 怎
			assertEquals(vocab.getId("么"), ids[6]);                     // 么
			assertEquals(vocab.getId("样"), ids[7]);                     // 样
			assertEquals(VocabLoader.SEP_ID, ids[8]);                    // [SEP]
		}

		@Test
		@DisplayName("「播放音乐」→ 逐字")
		void testPlayMusicSplit() {
			BertTokenizer.TokenResult result = tokenizer.tokenize("播放音乐");

			long[] ids = result.inputIds();
			assertEquals(VocabLoader.CLS_ID, ids[0]);
			assertEquals(vocab.getId("播"), ids[1]);
			assertEquals(vocab.getId("放"), ids[2]);
			assertEquals(vocab.getId("音"), ids[3]);
			assertEquals(vocab.getId("乐"), ids[4]);
			assertEquals(VocabLoader.SEP_ID, ids[5]);
		}

		@Test
		@DisplayName("中英文混合：逐字中文 + 连续英文")
		void testMixedCnEn() {
			// \u4F60\u597D = 你好, hello, \u4E16\u754C = 世界
			// 期望: [CLS] + 你 + 好 + hello + 世 + 界 + [SEP] = 7 real tokens
			BertTokenizer.TokenResult result = tokenizer.tokenize("\u4F60\u597Dhello\u4E16\u754C");

			long[] ids = result.inputIds();
			assertEquals(VocabLoader.CLS_ID, ids[0]);
			assertEquals(VocabLoader.SEP_ID, ids[6], "[SEP] should be at position 6");

			// 验证 attention_mask: 前 7 位为 1
			for (int i = 0; i < 7; i++) {
				assertEquals(1, result.attentionMask()[i],
					"attention_mask[" + i + "] should be 1");
			}
			assertEquals(0, result.attentionMask()[7], "position 7 should be padding");
		}

		@Test
		@DisplayName("纯英文：'hello world' → 按空格分词")
		void testPureEnglish() {
			BertTokenizer.TokenResult result = tokenizer.tokenize("hello world");

			long[] ids = result.inputIds();
			assertEquals(VocabLoader.CLS_ID, ids[0]);
			assertEquals(vocab.getId("hello"), ids[1]);
			assertEquals(vocab.getId("world"), ids[2]);
			assertEquals(VocabLoader.SEP_ID, ids[3]);
		}
	}

	// ==================== Padding 测试 ====================

	@Nested
	@DisplayName("Padding 和截断")
	class PaddingAndTruncation {

		@Test
		@DisplayName("短文本 padding 为 0")
		void testShortTextPadding() {
			int maxLen = 16;
			BertTokenizer tokenizer = new BertTokenizer(vocab, maxLen);
			BertTokenizer.TokenResult result = tokenizer.tokenize("你好"); // 2 CJK chars + CLS + SEP = 4 used

			long[] ids = result.inputIds();
			assertEquals(maxLen, ids.length, "inputIds 长度应等于 maxLength");
			assertEquals(VocabLoader.PAD_ID, ids[maxLen - 1], "末尾应为 [PAD]=0");

			// attention_mask: 前 4 个=1，其余=0
			assertEquals(1, result.attentionMask()[0]);
			assertEquals(1, result.attentionMask()[1]);
			assertEquals(1, result.attentionMask()[2]);
			assertEquals(1, result.attentionMask()[3]);
			assertEquals(0, result.attentionMask()[4]);
		}

		@Test
		@DisplayName("长文本截断：超过 maxLength 的 token 丢弃")
		void testLongTextTruncation() {
			int maxLen = 8; // CLS + 最多 6 token + SEP
			BertTokenizer tokenizer = new BertTokenizer(vocab, maxLen);

			// "今天天气怎么样播放音乐" = 10 个 CJK 字符
			String longText = "今天天气怎么样播放音乐";
			BertTokenizer.TokenResult result = tokenizer.tokenize(longText);

			assertEquals(maxLen, result.inputIds().length);
			assertEquals(VocabLoader.CLS_ID, result.inputIds()[0]);
			assertEquals(VocabLoader.SEP_ID, result.inputIds()[maxLen - 1],
				"截断后 [SEP] 应在末尾");
			assertEquals(1, result.attentionMask()[0]);
			assertEquals(1, result.attentionMask()[maxLen - 1],
				"[SEP] 的 attention_mask 始终为 1");
		}

		@Test
		@DisplayName("token_type_ids 全为 0（单句任务）")
		void testTokenTypeIdsAllZero() {
			BertTokenizer tokenizer = new BertTokenizer(vocab, 32);
			BertTokenizer.TokenResult result = tokenizer.tokenize("测试文本");

			for (int i = 0; i < result.tokenTypeIds().length; i++) {
				assertEquals(0, result.tokenTypeIds()[i],
					"token_type_ids[" + i + "] 应为 0");
			}
		}
	}

	// ==================== attention_mask 测试 ====================

	@Nested
	@DisplayName("attention_mask 正确性")
	class AttentionMask {

		@Test
		@DisplayName("实 token 为 1，padding 为 0")
		void testAttentionMaskValues() {
			int maxLen = 32;
			BertTokenizer tokenizer = new BertTokenizer(vocab, maxLen);
			BertTokenizer.TokenResult result = tokenizer.tokenize("你好世界");
			// CLS + 你 + 好 + 世 + 界 + SEP = 6 个实 token

			for (int i = 0; i < 6; i++) {
				assertEquals(1, result.attentionMask()[i],
					"前 6 个位置 attention_mask 应为 1");
			}
			for (int i = 6; i < maxLen; i++) {
				assertEquals(0, result.attentionMask()[i],
					"剩余 padding 位置 attention_mask 应为 0");
			}
		}
	}

	// ==================== 边界情况 ====================

	@Nested
	@DisplayName("边界情况")
	class EdgeCases {

		@Test
		@DisplayName("空文本：只有 [CLS] 和 [SEP]")
		void testEmptyText() {
			BertTokenizer tokenizer = new BertTokenizer(vocab, 32);
			BertTokenizer.TokenResult result = tokenizer.tokenize("");

			assertEquals(VocabLoader.CLS_ID, result.inputIds()[0]);
			assertEquals(VocabLoader.SEP_ID, result.inputIds()[1]);
			assertEquals(1, result.attentionMask()[0]);
			assertEquals(1, result.attentionMask()[1]);
			assertEquals(0, result.attentionMask()[2]);
		}

		@Test
		@DisplayName("null 文本：只有 [CLS] 和 [SEP]")
		void testNullText() {
			BertTokenizer tokenizer = new BertTokenizer(vocab, 32);
			BertTokenizer.TokenResult result = tokenizer.tokenize(null);

			assertEquals(VocabLoader.CLS_ID, result.inputIds()[0]);
			assertEquals(VocabLoader.SEP_ID, result.inputIds()[1]);
		}

		@Test
		@DisplayName("纯空白文本 → 跳过空白，只有 [CLS][SEP]")
		void testWhitespaceOnly() {
			BertTokenizer tokenizer = new BertTokenizer(vocab, 32);
			BertTokenizer.TokenResult result = tokenizer.tokenize("   \t  ");

			assertEquals(VocabLoader.CLS_ID, result.inputIds()[0]);
			assertEquals(VocabLoader.SEP_ID, result.inputIds()[1]);
			assertEquals(VocabLoader.PAD_ID, result.inputIds()[2]);
		}

		@Test
		@DisplayName("单个中文字符")
		void testSingleChineseChar() {
			BertTokenizer tokenizer = new BertTokenizer(vocab, 16);
			BertTokenizer.TokenResult result = tokenizer.tokenize("我");

			assertEquals(VocabLoader.CLS_ID, result.inputIds()[0]);
			assertEquals(vocab.getId("我"), result.inputIds()[1]);
			assertEquals(VocabLoader.SEP_ID, result.inputIds()[2]);
		}

		@Test
		@DisplayName("含标点的中文文本")
		void testChineseWithPunctuation() {
			BertTokenizer tokenizer = new BertTokenizer(vocab, 64);
			BertTokenizer.TokenResult result = tokenizer.tokenize("你好，世界！");

			assertEquals(VocabLoader.CLS_ID, result.inputIds()[0]);
			assertEquals(vocab.getId("你"), result.inputIds()[1]);
			assertEquals(vocab.getId("好"), result.inputIds()[2]);
			// 逗号 → 如果 vocab 有 ","，则返回其 ID，否则 UNK=100
			assertEquals(vocab.getId("世"), result.inputIds()[4]);
			assertEquals(vocab.getId("界"), result.inputIds()[5]);
			// 感叹号处理同理
		}
	}
}
