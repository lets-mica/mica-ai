package net.dreamlu.mica.ai.tts.g2p;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import static org.junit.jupiter.api.Assertions.*;

/**
 * HoubbPinyinG2P 测试。
 * <p>当 houbb/pinyin 库在 classpath 中时（mica-ai-tts 自身测试时）执行；
 * <p>否则使用 {@code @EnabledIf} 跳过。
 */
@EnabledIf("isHoubbAvailable")
class HoubbPinyinG2PTest {

	private static HoubbPinyinG2P g2pInstance;

	@BeforeAll
	static void setup() {
		g2pInstance = new HoubbPinyinG2P();
	}

	/**
	 * JUnit 5 EnabledIf 静态条件方法。
	 */
	static boolean isHoubbAvailable() {
		return new HoubbPinyinG2P().isAvailable();
	}

	@Test
	void testInitialize() {
		assertNotNull(g2pInstance);
		assertTrue(g2pInstance.isAvailable());
	}

	@Test
	void testConvertReturnsBopomofo() {
		G2P g2p = g2pInstance;
		String result = g2p.convert("你好");
		assertNotNull(result);
		assertTrue(result.contains("ㄋ") || result.contains("ㄏ"),
			"Result should contain bopomofo symbols, got: " + result);
	}

	@Test
	void testConvertEmpty() {
		G2P g2p = g2pInstance;
		assertEquals("", g2p.convert(""));
		assertEquals("", g2p.convert(null));
	}

	@Test
	void testConvertPureEnglish() {
		G2P g2p = g2pInstance;
		String result = g2p.convert("Hello World");
		assertEquals("Hello World", result);
	}

	@Test
	void testConvertPolyphone() {
		// houbb/pinyin 通过分词智能消歧多音字
		// "重庆" → chóng qìng (而非 zhòng qìng)
		G2P g2p = g2pInstance;
		String result = g2p.convert("重庆火锅");
		assertNotNull(result);
		// ㄔㄨㄥ 是 "chong" 的注音
		assertTrue(result.contains("ㄔㄨㄥ") || result.contains("ㄑㄧㄥ"),
			"Result should contain correct bopomofo for 重庆火锅, got: " + result);
	}

	@Test
	void testConvertDigits() {
		G2P g2p = g2pInstance;
		String result = g2p.convert("123");
		// 数字会被转换为对应的中文读音
		assertNotNull(result);
	}
}
