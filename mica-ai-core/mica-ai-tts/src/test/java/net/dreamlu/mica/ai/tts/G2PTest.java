package net.dreamlu.mica.ai.tts;

import net.dreamlu.mica.ai.tts.g2p.ChineseG2P;
import net.dreamlu.mica.ai.tts.g2p.ChineseTextNormalizer;
import net.dreamlu.mica.ai.tts.g2p.G2P;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * G2P 接口与默认实现测试。
 *
 * <p><b>声调标记约定</b>：</p>
 * <ul>
 *   <li>数字声调 1-5：{@link ChineseG2P} 字典文件 / 公开 API 使用（如 {@code ni3}）</li>
 *   <li>上标数字 ¹²³⁴⁵：{@link ChineseTextNormalizer} 内部使用（避开与阿拉伯数字冲突，
 *       不会触发 normalizeNumber 二次翻译）</li>
 * </ul>
 */
class G2PTest {

	// 上标数字常量
	private static final String T1 = "\u00B9"; // ¹
	private static final String T2 = "\u00B2"; // ²
	private static final String T3 = "\u00B3"; // ³
	private static final String T4 = "\u2074"; // ⁴
	private static final String T5 = "\u2075"; // ⁵

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
	void testPinyinToBopomofoSuperscriptTone() {
		// 上标数字声调（normalizer 使用），应与普通数字声调等价
		assertEquals("ㄋㄧ", ChineseG2P.pinyinToBopomofo("ni" + T3));
		assertEquals("ㄏㄠ", ChineseG2P.pinyinToBopomofo("hao" + T3));
		assertEquals("ㄓㄨㄥ", ChineseG2P.pinyinToBopomofo("zhong" + T1));
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
		assertTrue(result.contains("世") || result.contains("ㄕ"),
			"Result should contain Chinese bopomofo");
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

	// ========================================================================
	// 新功能：Heteronym 行内标记（$行=hang2$）
	// ========================================================================

	@Test
	void testHeteronymInline() {
		// 行内标记强制覆盖多音字
		// 默认 行 读 xing2；标记后强制为 hang2
		G2P g2p = ChineseG2P.getDefault();
		String result = g2p.convert("银行的$行=hang2$长很行");
		assertNotNull(result);
		// hang2 → ㄏㄤ（hang 是 ang 韵母）
		assertTrue(result.contains("ㄏㄤ"),
			"Heteronym should force 行 to read hang2 (ㄏㄤ), got: " + result);
	}

	@Test
	void testHeteronymSessionOverride() {
		// 会话级 API
		ChineseG2P g2p = ChineseG2P.getDefault().withHeteronym('行', "hang2");
		String result = g2p.convert("行长");
		assertNotNull(result);
		assertTrue(result.contains("ㄏㄤ"),
			"withHeteronym should override 行 to hang2, got: " + result);
	}

	@Test
	void testHeteronymIsImmutable() {
		// 链式调用应该返回新实例，不修改原单例
		ChineseG2P base = ChineseG2P.getDefault();
		ChineseG2P overridden = base.withHeteronym('行', "hang2");
		assertNotSame(base, overridden);
		// 原实例行为不变
		String baseResult = base.convert("行");
		assertFalse(baseResult.contains("ㄏㄤ"),
			"Base instance should not be affected by withHeteronym");
	}

	// ========================================================================
	// 新功能：文本归一化（数字、金额、日期、时间）
	// 注：normalizer 输出使用上标数字声调 ¹²³⁴⁵（避开阿拉伯数字与 normalizer 二次翻译冲突）
	// ========================================================================

	@Test
	void testNormalizeDigit() {
		assertEquals("yi" + T1, ChineseTextNormalizer.normalize("1").trim());
		assertEquals("shi" + T2, ChineseTextNormalizer.normalize("10").trim());
		assertEquals("yi" + T1 + " bai" + T3 + " er" + T4 + " shi" + T2 + " san" + T1,
			ChineseTextNormalizer.normalize("123").trim());
		assertEquals("yi" + T1 + " qian" + T1 + " er" + T4 + " bai" + T3 + " ling" + T5 + " san" + T1,
			ChineseTextNormalizer.normalize("1203").trim());
	}

	@Test
	void testNormalizeDecimal() {
		String r = ChineseTextNormalizer.normalize("3.14");
		assertTrue(r.contains("san" + T1) && r.contains("dian" + T3)
			&& r.contains("yi" + T1) && r.contains("si" + T4),
			"3.14 should be 三点一四, got: " + r);
	}

	@Test
	void testNormalizeNegative() {
		String r = ChineseTextNormalizer.normalize("-5");
		assertTrue(r.contains("fu" + T4) && r.contains("wu" + T3),
			"-5 should be 负五, got: " + r);
	}

	@Test
	void testNormalizePercent() {
		String r = ChineseTextNormalizer.normalize("50%");
		assertTrue(r.contains("bai" + T3) && r.contains("fen" + T1)
			&& r.contains("zhi" + T1) && r.contains("wu" + T3),
			"50% should be 百分之五十, got: " + r);
	}

	@Test
	void testNormalizeCurrencyCN() {
		String r = ChineseTextNormalizer.normalize("¥128");
		assertTrue(r.contains("yuan" + T2), "¥128 should mention 元, got: " + r);
		assertTrue(r.contains("yi" + T1) && r.contains("er" + T4)
			&& r.contains("shi" + T2) && r.contains("ba" + T1),
			"¥128 should mention 一二八, got: " + r);
	}

	@Test
	void testNormalizeYearMonth() {
		String r = ChineseTextNormalizer.normalize("2026年");
		// 2026年 → 二零二六年（5 syllables）
		assertTrue(r.contains("er" + T4) && r.contains("ling" + T5)
			&& r.contains("liu" + T4) && r.contains("nian" + T2),
			"2026年 should be 二零二六年, got: " + r);
	}

	@Test
	void testNormalizeTime() {
		String r = ChineseTextNormalizer.normalize("12:34");
		assertTrue(r.contains("shi" + T2) && r.contains("er" + T4) && r.contains("dian" + T3),
			"12:34 should be 十二点, got: " + r);
	}

	@Test
	void testNormalizePhone() {
		String r = ChineseTextNormalizer.normalize("13800138000");
		assertTrue(r.contains("yi" + T1) && r.contains("san" + T1)
			&& r.contains("ba" + T1) && r.contains("ling" + T5),
			"Phone digits should be read one by one, got: " + r);
	}

	@Test
	void testNormalizeEmpty() {
		assertEquals("", ChineseTextNormalizer.normalize(null));
		assertEquals("", ChineseTextNormalizer.normalize(""));
	}

	// ========================================================================
	// 新功能：英文逐字母读音
	// ========================================================================

	@Test
	void testEnglishByLetter() {
		G2P g2p = ChineseG2P.getDefault();
		String result = g2p.convert("API");
		assertNotNull(result);
		// A → ei4, P → pi4, I → ai4
		assertTrue(result.contains("ei4") && result.contains("pi4") && result.contains("ai4"),
			"API should be read A-P-I (ei4 pi4 ai4), got: " + result);
	}

	@Test
	void testEnglishLookup() {
		G2P g2p = ChineseG2P.getDefault();
		String result = g2p.convert("OK");
		assertNotNull(result);
		// 命中 ENGLISH_LOOKUP → "ou1 kei4"
		assertTrue(result.contains("ou1") && result.contains("kei4"),
			"OK should be ou1 kei4, got: " + result);
	}

	@Test
	void testMixedChineseEnglish() {
		G2P g2p = ChineseG2P.getDefault();
		String result = g2p.convert("我爱你China");
		assertNotNull(result);
		// 中文 → 注音；China → 字母串读
		assertTrue(result.contains("ㄨㄛ") && result.contains("ㄞ") && result.contains("ㄋㄧ"),
			"Should contain Chinese bopomofo, got: " + result);
		assertTrue(result.contains("ei4") || result.contains("xi4"),
			"Should contain English letter pronunciation, got: " + result);
	}
}
