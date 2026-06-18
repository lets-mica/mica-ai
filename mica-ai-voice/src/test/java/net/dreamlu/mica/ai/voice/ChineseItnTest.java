package net.dreamlu.mica.ai.voice;

import net.dreamlu.mica.ai.voice.internal.ChineseItn;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * ChineseItn 中文数字转阿拉伯数字测试。
 */
class ChineseItnTest {

	@Test
	void testPureNumber() {
		assertEquals("192", ChineseItn.convert("幺九二").trim());
	}

	@Test
	void testValueNumber() {
		String result = ChineseItn.convert("二十五");
		assertTrue(result.contains("25"), "Expected 25, got: " + result);
	}

	@Test
	void testHundred() {
		String result = ChineseItn.convert("三百");
		assertTrue(result.contains("300"), "Expected 300, got: " + result);
	}

	@Test
	void testIdiom() {
		// 成语不应被转换
		String result = ChineseItn.convert("乱七八糟");
		assertEquals("乱七八糟", result);
	}

	@Test
	void testEmptyAndNull() {
		assertNull(ChineseItn.convert(null));
		assertEquals("", ChineseItn.convert(""));
	}

	@Test
	void testNoChineseNumber() {
		assertEquals("你好世界", ChineseItn.convert("你好世界"));
	}
}
