package net.dreamlu.mica.ai.tts;

import net.dreamlu.mica.ai.tts.g2p.G2P;
import net.dreamlu.mica.ai.tts.g2p.ChineseG2P;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * G2P 接口与默认实现测试。
 */
class G2PTest {

	@Test
	void testChineseG2PGetDefault() {
		G2P g2p1 = ChineseG2P.getDefault();
		G2P g2p2 = ChineseG2P.getDefault();
		assertSame(g2p1, g2p2, "Should return singleton instance");
	}

	@Test
	void testChineseG2PImplementsInterface() {
		assertTrue(G2P.class.isAssignableFrom(ChineseG2P.class),
			"ChineseG2P should implement G2P interface");
	}

	@Test
	void testPinyinToBopomofoBasic() {
		assertEquals("ㄋㄧ", ChineseG2P.pinyinToBopomofo("ni3"));
		assertEquals("ㄏㄠ", ChineseG2P.pinyinToBopomofo("hao3"));
		assertEquals("ㄓㄨㄥ", ChineseG2P.pinyinToBopomofo("zhong1"));
	}

	@Test
	void testPinyinToBopomofoEmpty() {
		assertEquals("", ChineseG2P.pinyinToBopomofo(""));
		assertEquals("", ChineseG2P.pinyinToBopomofo(null));
	}

	@Test
	void testPinyinToBopomofoSpecial() {
		// zh ch sh 双字母声母
		assertEquals("ㄓ", ChineseG2P.pinyinToBopomofo("zhi1").substring(0, 1));
		assertEquals("ㄔ", ChineseG2P.pinyinToBopomofo("chi1").substring(0, 1));
		assertEquals("ㄕ", ChineseG2P.pinyinToBopomofo("shi1").substring(0, 1));
	}

	@Test
	void testConvertChinese() {
		G2P g2p = ChineseG2P.getDefault();
		String result = g2p.convert("你好");
		assertNotNull(result);
		assertTrue(result.contains("ㄋ") || result.contains("ㄏ"),
			"Result should contain bopomofo for 你 or 好");
	}

	@Test
	void testConvertMixed() {
		G2P g2p = ChineseG2P.getDefault();
		String result = g2p.convert("Hello 世界");
		assertNotNull(result);
		assertTrue(result.contains("Hello"));
	}

	@Test
	void testConvertEmpty() {
		G2P g2p = ChineseG2P.getDefault();
		assertEquals("", g2p.convert(""));
		assertEquals("", g2p.convert(null));
	}

	@Test
	void testConvertUnknownChars() {
		// 未知汉字应被丢弃，不污染音素序列
		G2P g2p = ChineseG2P.getDefault();
		String result = g2p.convert("龘"); // 罕见字
		assertEquals("", result);
	}

	@Test
	void testCustomG2PLambda() {
		// 测试 G2P 作为函数式接口：使用 lambda 注入
		G2P customG2p = text -> "test_" + text;
		assertEquals("test_hello", customG2p.convert("hello"));
	}
}
