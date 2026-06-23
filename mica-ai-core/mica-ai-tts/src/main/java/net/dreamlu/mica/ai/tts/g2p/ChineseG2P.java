package net.dreamlu.mica.ai.tts.g2p;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * 简体中文 G2P 转换器（零依赖 fallback 实现）。
 *
 * <p>基于内置汉字-拼音字典 + 常用词组表实现，输出注音符号（Bopomofo）。</p>
 *
 * <p>字典来源：</p>
 * <ul>
 *   <li>单字字典：{@code classpath:tts/chinese-char-pinyin.txt}（~3500 常用汉字）</li>
 *   <li>词组字典：{@code classpath:tts/chinese-word-pinyin.txt}（~5000 常用词组，处理多音字消歧）</li>
 * </ul>
 *
 * <p>字典文件格式（每行一项，UTF-8）：</p>
 * <pre>
 * 字=pinyin           # 单字，可选多音字用 / 分隔
 * 词语=pinyin1 pinyin2  # 词组（2-4 字），拼音之间用空格分隔
 * </pre>
 *
 * <p><b>生产环境推荐使用 {@link HoubbPinyinG2P}</b>（基于 houbb/pinyin，7 万+ 字符 + 多音字智能消歧）。
 * 本类主要作为零依赖降级方案 + 离线测试使用。</p>
 *
 * @author L.cm
 */
public final class ChineseG2P implements G2P {

	// ----------------------------------------------------------------
	// 声母 → 注音符号映射（按 key 长度倒序匹配，zh/ch/sh 双字母优先）
	// ----------------------------------------------------------------
	private static final String[] INITIAL_KEYS = {
		"zh", "ch", "sh",
		"b", "p", "m", "f",
		"d", "t", "n", "l",
		"g", "k", "h",
		"j", "q", "x",
		"r", "z", "c", "s",
		"y", "w"
	};
	private static final Map<String, String> INITIALS = new HashMap<>();
	static {
		INITIALS.put("zh", "ㄓ"); INITIALS.put("ch", "ㄔ"); INITIALS.put("sh", "ㄕ");
		INITIALS.put("b", "ㄅ"); INITIALS.put("p", "ㄆ"); INITIALS.put("m", "ㄇ"); INITIALS.put("f", "ㄈ");
		INITIALS.put("d", "ㄉ"); INITIALS.put("t", "ㄊ"); INITIALS.put("n", "ㄋ"); INITIALS.put("l", "ㄌ");
		INITIALS.put("g", "ㄍ"); INITIALS.put("k", "ㄎ"); INITIALS.put("h", "ㄏ");
		INITIALS.put("j", "ㄐ"); INITIALS.put("q", "ㄑ"); INITIALS.put("x", "ㄒ");
		INITIALS.put("r", "ㄖ"); INITIALS.put("z", "ㄗ"); INITIALS.put("c", "ㄘ"); INITIALS.put("s", "ㄙ");
		INITIALS.put("y", ""); INITIALS.put("w", "");
	}

	// 韵母 → 注音符号映射（按 key 长度倒序匹配）
	private static final String[] FINAL_KEYS = {
		"iang", "iong", "uang", "ueng",
		"uai", "uan", "ian", "iao", "iang",
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

	// 数字 → 拼音
	private static final Map<Character, String> NUMBER_MAP = new HashMap<>();
	static {
		NUMBER_MAP.put('0', "ling2"); NUMBER_MAP.put('1', "yi1");
		NUMBER_MAP.put('2', "er4"); NUMBER_MAP.put('3', "san1");
		NUMBER_MAP.put('4', "si4"); NUMBER_MAP.put('5', "wu3");
		NUMBER_MAP.put('6', "liu4"); NUMBER_MAP.put('7', "qi1");
		NUMBER_MAP.put('8', "ba1"); NUMBER_MAP.put('9', "jiu3");
	}

	// 字典路径
	private static final String CHAR_DICT_PATH = "/tts/chinese-char-pinyin.txt";
	private static final String WORD_DICT_PATH = "/tts/chinese-word-pinyin.txt";

	private static volatile Map<String, String> CHAR_DICT;
	private static volatile Map<String, String> WORD_DICT;

	private static volatile ChineseG2P INSTANCE;

	/**
	 * 获取默认实例（单例，延迟初始化）。
	 */
	public static ChineseG2P getDefault() {
		ChineseG2P g = INSTANCE;
		if (g == null) {
			synchronized (ChineseG2P.class) {
				g = INSTANCE;
				if (g == null) {
					g = new ChineseG2P();
					INSTANCE = g;
				}
			}
		}
		return g;
	}

	/**
	 * 将拼音（如 "ni3", "zhong1"）转换为注音符号。
	 * <p>该方法公开，方便其它 G2P 实现复用。</p>
	 *
	 * @param pinyin 拼音（带数字声调，1-5，5=轻声）
	 * @return 注音符号字符串（如 "ni3" → "ㄋㄧ"）
	 */
	public static String pinyinToBopomofo(String pinyin) {
		if (pinyin == null || pinyin.isEmpty()) return "";
		String base = pinyin.replaceAll("[1-5]", "");
		if (base.isEmpty()) return "";

		// 提取声母（长 key 优先）
		String initial = "";
		String rest = base;
		for (String ini : INITIAL_KEYS) {
			if (base.startsWith(ini)) {
				initial = INITIALS.get(ini);
				rest = base.substring(ini.length());
				break;
			}
		}
		// y/w 开头的特殊处理：y → i (yi/ya/yan/you)，yu → v (yuan/yue)，w → u (wa/wu)
		if (initial.isEmpty() && base.length() > 1) {
			char head = base.charAt(0);
			if (head == 'y') {
				if (base.length() > 2 && base.charAt(1) == 'u') {
					rest = "v" + base.substring(2);
				} else {
					rest = base.substring(1);
				}
			} else if (head == 'w') {
				rest = base.substring(1);
			}
		}

		// 查韵母（长 key 优先）
		String finalBpmf = "";
		for (String f : FINAL_KEYS) {
			if (rest.startsWith(f)) {
				finalBpmf = FINALS.get(f);
				break;
			}
		}
		return initial + finalBpmf;
	}

	@Override
	public String convert(String text) {
		if (text == null || text.isEmpty()) return "";
		Map<String, String> charDict = charDict();
		Map<String, String> wordDict = wordDict();

		StringBuilder sb = new StringBuilder();
		int i = 0;
		int n = text.length();
		while (i < n) {
			char c = text.charAt(i);
			// 1. 优先匹配词组（最长 4 字）
			boolean matchedWord = false;
			for (int len = 4; len >= 2; len--) {
				if (i + len <= n) {
					String word = text.substring(i, i + len);
					String pinyinSeq = wordDict.get(word);
					if (pinyinSeq != null) {
						appendPinyinSequence(sb, pinyinSeq);
						i += len;
						matchedWord = true;
						break;
					}
				}
			}
			if (matchedWord) continue;

			// 2. 单字处理
			if (isChinese(c)) {
				String pinyin = lookupCharPinyin(c, charDict);
				if (pinyin != null && !pinyin.isEmpty()) {
					String bpmf = pinyinToBopomofo(pinyin);
					appendWithSpace(sb, bpmf);
				}
				i++;
			} else if (Character.isDigit(c)) {
				String pinyin = NUMBER_MAP.get(c);
				if (pinyin != null) {
					String bpmf = pinyinToBopomofo(pinyin);
					appendWithSpace(sb, bpmf);
				}
				i++;
			} else if (isEnglishLetter(c)) {
				sb.append(c);
				i++;
			} else if (isPunctuation(c)) {
				sb.append(c);
				i++;
			} else if (Character.isWhitespace(c)) {
				if (!sb.isEmpty() && sb.charAt(sb.length() - 1) != ' ') {
					sb.append(' ');
				}
				i++;
			} else {
				i++;
			}
		}
		return sb.toString().trim();
	}

	private static void appendPinyinSequence(StringBuilder sb, String pinyinSeq) {
		String[] syllables = pinyinSeq.split("\\s+");
		for (String s : syllables) {
			if (!s.isEmpty()) {
				String bpmf = pinyinToBopomofo(s);
				appendWithSpace(sb, bpmf);
			}
		}
	}

	private static String lookupCharPinyin(char c, Map<String, String> dict) {
		String pinyinSeq = dict.get(String.valueOf(c));
		if (pinyinSeq == null) return null;
		// 多音字取第一个
		int slash = pinyinSeq.indexOf('/');
		return slash < 0 ? pinyinSeq : pinyinSeq.substring(0, slash);
	}

	private static void appendWithSpace(StringBuilder sb, String s) {
		if (s == null || s.isEmpty()) return;
		if (!sb.isEmpty() && sb.charAt(sb.length() - 1) != ' ') {
			sb.append(' ');
		}
		sb.append(s);
	}

	private static boolean isChinese(char c) {
		return c >= '\u4e00' && c <= '\u9fff';
	}

	private static boolean isEnglishLetter(char c) {
		return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
	}

	private static boolean isPunctuation(char c) {
		return ",.!?;:，。！？；：、（）()\"'\"'—…《》<>【】".indexOf(c) >= 0;
	}

	// ========================================================================
	// 字典加载（首次访问时按需懒加载，线程安全）
	// ========================================================================

	private static Map<String, String> charDict() {
		Map<String, String> d = CHAR_DICT;
		if (d != null) return d;
		synchronized (ChineseG2P.class) {
			d = CHAR_DICT;
			if (d == null) {
				d = loadDict(CHAR_DICT_PATH);
				CHAR_DICT = d;
			}
		}
		return d;
	}

	private static Map<String, String> wordDict() {
		Map<String, String> d = WORD_DICT;
		if (d != null) return d;
		synchronized (ChineseG2P.class) {
			d = WORD_DICT;
			if (d == null) {
				d = loadDict(WORD_DICT_PATH);
				WORD_DICT = d;
			}
		}
		return d;
	}

	private static Map<String, String> loadDict(String classpathPath) {
		Map<String, String> m = new HashMap<>(4096);
		InputStream in = ChineseG2P.class.getResourceAsStream(classpathPath);
		if (in == null) {
			// 字典文件缺失，返回空字典（G2P 对未知汉字降级为丢弃）
			return m;
		}
		try (BufferedReader reader = new BufferedReader(
			new InputStreamReader(in, StandardCharsets.UTF_8))) {
			String line;
			while ((line = reader.readLine()) != null) {
				line = line.trim();
				if (line.isEmpty() || line.startsWith("#")) continue;
				int eq = line.indexOf('=');
				if (eq <= 0) continue;
				String key = line.substring(0, eq).trim();
				String value = line.substring(eq + 1).trim();
				if (!key.isEmpty() && !value.isEmpty()) {
					m.put(key, value);
				}
			}
		} catch (Exception e) {
			// 静默降级
		}
		return m;
	}
}
