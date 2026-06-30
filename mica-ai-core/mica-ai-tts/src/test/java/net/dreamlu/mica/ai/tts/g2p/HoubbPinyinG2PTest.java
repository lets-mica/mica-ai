package net.dreamlu.mica.ai.tts.g2p;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * HoubbPinyinG2P 测试。
 */
class HoubbPinyinG2PTest {

	private final G2P g2p = new HoubbPinyinG2P();

	@Test
	void testConvertReturnsBopomofo() {
		String result = g2p.convert("你好");
		assertNotNull(result);
		assertTrue(result.contains("ㄋ") || result.contains("ㄏ"),
			"Result should contain bopomofo symbols, got: " + result);
	}

	@Test
	void testConvertEmpty() {
		assertEquals("", g2p.convert(""));
		assertEquals("", g2p.convert(null));
	}

	@Test
	void testConvertPureEnglish() {
		// 不在 IPA 字典中的纯英文：letter-by-letter 展开，每个字母后接声调数字 '1'
		String result = g2p.convert("Hello World");
		assertEquals("H1 e1 l1 l1 o1 W1 o1 r1 l1 d1", result);
	}

	@Test
	void testConvertMicaAI() {
		// mica-ai 在内置 IPA 字典中 → 输出完整 IPA 音素序列
		String result = g2p.convert("mica-ai");
		// m + aɪ (双元音 a + ɪ) + k + ə + 主重音 ˈ + aɪ
		assertEquals("m aɪ k ə ˈ aɪ", result);
	}

	@Test
	void testConvertMixedChineseEnglish() {
		// 中文 + 字典命中英文：中文走 Bopomofo，mica-ai 走 IPA 字典
		String result = g2p.convert("你好，欢迎使用 mica-ai。");
		// 验证 mica-ai 部分被替换为 IPA 音素
		assertTrue(result.endsWith("m aɪ k ə ˈ aɪ .") || result.endsWith("m aɪ k ə ˈ aɪ"),
			"mica-ai should be replaced by IPA phonemes, got: " + result);
		// 验证中文部分有 Bopomofo
		assertTrue(result.contains("ㄋ") || result.contains("ㄏ"),
			"Chinese part should contain bopomofo, got: " + result);
	}

	@Test
	void testConvertMicaAICaseInsensitive() {
		// 字典 key 小写匹配：不区分大小写
		String upper = g2p.convert("MICA-AI");
		String lower = g2p.convert("mica-ai");
		assertEquals(lower, upper);
		assertEquals("m aɪ k ə ˈ aɪ", upper);
	}

	@Test
	void testConvertPolyphone() {
		// houbb/pinyin 通过分词智能消歧多音字
		// "重庆" → chóng qìng (而非 zhòng qìng)
		String result = g2p.convert("重庆火锅");
		assertNotNull(result);
		// ㄔㄨㄥ 是 "chong" 的注音
		assertTrue(result.contains("ㄔㄨㄥ") || result.contains("ㄑㄧㄥ"),
			"Result should contain correct bopomofo for 重庆火锅, got: " + result);
	}

	@Test
	void testConvertDigits() {
		String result = g2p.convert("123");
		// 数字会被转换为对应的中文读音
		assertNotNull(result);
	}
}
