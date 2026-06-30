package net.dreamlu.mica.ai.tts.g2p;

import java.util.HashMap;
import java.util.Map;

/**
 * 拼音 → 注音符号（bopomofo）转换器。
 *
 * <p>基于拼音/声母/韵母的静态映射，支持数字声调（1-5）和上标数字声调（¹²³⁴⁵）。</p>
 *
 * <p>与 Kokoro-82M 训练数据的注音体系一致：
 * <ul>
 *     <li>声母 bpmf：ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ</li>
 *     <li>韵母 bpmf：ㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ + ㄧㄨㄩ 系列（i/ia/ie/ian/iang/iao/in/ing/iong/iu/u/ua/uai/uan/uang/ue/ui/uo/un/uo/v/ve）</li>
 * </ul>
 *
 * <p>特殊处理：
 * <ul>
 *     <li><b>y 头韵母</b>：yi/yan/ying/yong 等 → 走 i 系列韵母（不预设声母 ㄧ）</li>
 *     <li><b>w 头韵母</b>：wa/wan/wang/wo 等 → 走 u 系列韵母</li>
 *     <li><b>yu</b> → ㄩ；<b>wu</b> → ㄨ</li>
 * </ul>
 *
 * @author L.cm
 */
public final class BopomofoConverter {

	private BopomofoConverter() {}

	// 声母（双字母优先）
	private static final String[] INITIAL_KEYS = {
		"zh", "ch", "sh",
		"b", "p", "m", "f",
		"d", "t", "n", "l",
		"g", "k", "h",
		"j", "q", "x",
		"r", "z", "c", "s"
	};
	private static final Map<String, String> INITIALS = new HashMap<>();
	static {
		INITIALS.put("zh", "ㄓ"); INITIALS.put("ch", "ㄔ"); INITIALS.put("sh", "ㄕ");
		INITIALS.put("b", "ㄅ"); INITIALS.put("p", "ㄆ"); INITIALS.put("m", "ㄇ"); INITIALS.put("f", "ㄈ");
		INITIALS.put("d", "ㄉ"); INITIALS.put("t", "ㄊ"); INITIALS.put("n", "ㄋ"); INITIALS.put("l", "ㄌ");
		INITIALS.put("g", "ㄍ"); INITIALS.put("k", "ㄎ"); INITIALS.put("h", "ㄏ");
		INITIALS.put("j", "ㄐ"); INITIALS.put("q", "ㄑ"); INITIALS.put("x", "ㄒ");
		INITIALS.put("r", "ㄖ"); INITIALS.put("z", "ㄗ"); INITIALS.put("c", "ㄘ"); INITIALS.put("s", "ㄙ");
	}

	// 韵母（长 key 优先）
	private static final String[] FINAL_KEYS = {
		"iang", "iong", "uang", "ueng",
		"uai", "uan", "ian", "iao",
		"ang", "eng", "ing", "ong",
		"ai", "ei", "ui", "ao", "ou", "iu",
		"ie", "ve", "er",
		"an", "en", "in", "un", "vn", "van",
		"ia", "ua", "uo",
		"a", "o", "e", "i", "u", "v"
	};
	private static final Map<String, String> FINALS = new HashMap<>();
	static {
		FINALS.put("iang", "ㄧㄤ"); FINALS.put("iong", "ㄩㄥ");
		FINALS.put("uang", "ㄨㄤ"); FINALS.put("ueng", "ㄨㄥ");
		FINALS.put("uai", "ㄨㄞ"); FINALS.put("uan", "ㄨㄢ");
		FINALS.put("ian", "ㄧㄢ"); FINALS.put("iao", "ㄧㄠ");
		FINALS.put("ang", "ㄤ"); FINALS.put("eng", "ㄥ");
		FINALS.put("ing", "ㄧㄥ"); FINALS.put("ong", "ㄨㄥ");
		FINALS.put("ai", "ㄞ"); FINALS.put("ei", "ㄟ"); FINALS.put("ui", "ㄨㄟ");
		FINALS.put("ao", "ㄠ"); FINALS.put("ou", "ㄡ"); FINALS.put("iu", "ㄧㄡ");
		FINALS.put("ie", "ㄧㄝ"); FINALS.put("ve", "ㄩㄝ"); FINALS.put("er", "ㄦ");
		FINALS.put("an", "ㄢ"); FINALS.put("en", "ㄣ"); FINALS.put("in", "ㄧㄣ"); FINALS.put("un", "ㄨㄣ");
		FINALS.put("vn", "ㄩㄣ"); FINALS.put("van", "ㄩㄢ");
		FINALS.put("ia", "ㄧㄚ"); FINALS.put("ua", "ㄨㄚ"); FINALS.put("uo", "ㄨㄛ");
		FINALS.put("a", "ㄚ"); FINALS.put("o", "ㄛ"); FINALS.put("e", "ㄜ");
		FINALS.put("i", "ㄧ"); FINALS.put("u", "ㄨ"); FINALS.put("v", "ㄩ");
	}

	// y 头拼音 → 标准拼音（用于重新走声母+韵母匹配路径）
	private static final Map<String, String> Y_HEAD = new HashMap<>();
	static {
		// 完整字直接重映射，避免 "yi" → "ii" / "yi" → "i" + "i" 这类多 i / 多 u 问题
		Y_HEAD.put("yi", "i");
		Y_HEAD.put("yin", "in");
		Y_HEAD.put("ying", "ing");
		Y_HEAD.put("ya", "ia");
		Y_HEAD.put("yan", "ian");
		Y_HEAD.put("yang", "iang");
		Y_HEAD.put("yao", "iao");
		Y_HEAD.put("ye", "ie");
		Y_HEAD.put("yo", "io");
		Y_HEAD.put("yong", "iong");
		Y_HEAD.put("you", "iu");
		// yu 系列 → ㄩ
		Y_HEAD.put("yu", "v");
		Y_HEAD.put("yuan", "van");
		Y_HEAD.put("yue", "ve");
		Y_HEAD.put("yun", "vn");
	}

	// w 头拼音 → 标准拼音
	private static final Map<String, String> W_HEAD = new HashMap<>();
	static {
		// 完整字直接重映射，避免 "wen" → "uen" / "wei" → "uei" 这类错误
		W_HEAD.put("wa", "ua");
		W_HEAD.put("wai", "uai");
		W_HEAD.put("wan", "uan");
		W_HEAD.put("wang", "uang");
		W_HEAD.put("wei", "ui");
		W_HEAD.put("wen", "un");
		W_HEAD.put("weng", "ueng");
		W_HEAD.put("wo", "uo");
		// wu 走零韵母，特殊处理
	}

	/**
	 * 拼音 → 注音符号。
	 *
	 * <p>支持三种声调标记：
	 * <ul>
	 *     <li>数字声调：1-5（5=轻声），如 {@code "ni3"}</li>
	 *     <li>上标数字：¹²³⁴⁵（Unicode 00B9/00B2/00B3/2074/2075），如 {@code "ni³"}</li>
	 * </ul>
	 *
	 * @param pinyin 拼音（带声调标记或不带）
	 * @return 注音符号字符串，如 {@code "ni3" → "ㄋㄧ"}；空串或 null 返回 {@code ""}
	 */
	public static String convert(String pinyin) {
		if (pinyin == null || pinyin.isEmpty()) return "";
		String base = pinyin
			.replaceAll("[1-5]", "")
			.replaceAll("[\u00B9\u00B2\u00B3\u2074\u2075]", "");
		if (base.isEmpty()) return "";

		// 1. 标准声母匹配
		String initial = "";
		String rest = base;
		for (String ini : INITIAL_KEYS) {
			if (base.startsWith(ini)) {
				initial = INITIALS.get(ini);
				rest = base.substring(ini.length());
				break;
			}
		}

		// 2. y/w 头拼音重映射（避免 yi/yin/ying → ii/iin/iing 这类多 i 问题）
		if (initial.isEmpty() && base.length() > 1) {
			char head = base.charAt(0);
			if (head == 'y') {
				// 查表：yi/yin/ying/ya/yan/yang/yao/ye/yo/yong/you/yu/yuan/yue/yun
				String mapped = Y_HEAD.get(base);
				if (mapped != null) {
					if (mapped.startsWith("v")) {
						// yu/yuan/yue/yun → 走 ㄩ 声母
						initial = "ㄩ";
						rest = mapped.substring(1);
					} else {
						// ya/yan/yang/... → 走 i 韵母
						initial = "";
						rest = mapped;
					}
				}
			} else if (head == 'w') {
				if (base.length() == 2 && base.charAt(1) == 'u') {
					// wu → ㄨ
					initial = "ㄨ";
					rest = "";
				} else {
					String mapped = W_HEAD.get(base);
					if (mapped != null) {
						initial = "";
						rest = mapped;
					}
				}
			}
		}

		// 3. 韵母匹配
		String finalBpmf = "";
		for (String f : FINAL_KEYS) {
			if (rest.startsWith(f)) {
				finalBpmf = FINALS.get(f);
				break;
			}
		}
		return initial + finalBpmf;
	}
}
