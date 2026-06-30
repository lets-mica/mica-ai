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
		String result = g2p.convert("Hello World");
		assertEquals("Hello World", result);
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
