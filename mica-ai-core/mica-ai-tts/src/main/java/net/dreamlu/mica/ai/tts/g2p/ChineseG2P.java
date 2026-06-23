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
 * <h3>处理流程</h3>
 * <ol>
 *   <li>{@link ChineseTextNormalizer} 文本归一化（数字、金额、日期、时间、百分号、电话）</li>
 *   <li>Heteronym 行内标记（{@code $行=hang2$} 强制多音字读音）</li>
 *   <li>正则分段：CJK 段走词组/单字查表；非 CJK 段处理英文（逐字母）/ 标点直通</li>
 *   <li>拼音 → 注音符号（bopomofo）映射</li>
 * </ol>
 *
 * <h3>字典来源</h3>
 * <ul>
 *   <li>单字：{@code classpath:tts/chinese-char-pinyin.txt}（~2000 常用字）</li>
 *   <li>词组：{@code classpath:tts/chinese-word-pinyin.txt}（~500 多音字消歧）</li>
 * </ul>
 *
 * <p><b>生产环境推荐 {@link HoubbPinyinG2P}</b>（基于 houbb/pinyin，7 万+ 字符 + 智能分词消歧）。
 * 本类作为零依赖降级方案 + 离线测试。</p>
 *
 * @author L.cm
 */
public final class ChineseG2P implements G2P {

	// ----------------------------------------------------------------
	// 声母（zh/ch/sh 双字母优先）
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
		INITIALS.put("y", "ㄧ"); INITIALS.put("w", "ㄨ");
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

	// 数字 → 拼音（备用，主要归一化由 ChineseTextNormalizer 处理）
	private static final Map<Character, String> NUMBER_MAP = new HashMap<>();
	static {
		NUMBER_MAP.put('0', "ling2"); NUMBER_MAP.put('1', "yi1");
		NUMBER_MAP.put('2', "er4"); NUMBER_MAP.put('3', "san1");
		NUMBER_MAP.put('4', "si4"); NUMBER_MAP.put('5', "wu3");
		NUMBER_MAP.put('6', "liu4"); NUMBER_MAP.put('7', "qi1");
		NUMBER_MAP.put('8', "ba1"); NUMBER_MAP.put('9', "jiu3");
	}

	// 英文 → 中文（一些常用的英文词 / 缩写，按字面意思读）
	private static final Map<String, String> ENGLISH_LOOKUP = new HashMap<>();
	static {
		ENGLISH_LOOKUP.put("OK", "ou1 kei4");
		ENGLISH_LOOKUP.put("ok", "ou1 kei4");
		ENGLISH_LOOKUP.put("NO", "en4 ou1");
		ENGLISH_LOOKUP.put("no", "en4 ou1");
		ENGLISH_LOOKUP.put("YES", "ye4 si1");
		ENGLISH_LOOKUP.put("yes", "ye4 si1");
		ENGLISH_LOOKUP.put("Hi", "hai1");
		ENGLISH_LOOKUP.put("hi", "hai1");
		ENGLISH_LOOKUP.put("Bye", "bai4");
		ENGLISH_LOOKUP.put("bye", "bai4");
		ENGLISH_LOOKUP.put("Hello", "he1 lo4");
		ENGLISH_LOOKUP.put("hello", "he1 lo4");
		ENGLISH_LOOKUP.put("ByeBye", "bai4 bai4");
		ENGLISH_LOOKUP.put("byebye", "bai4 bai4");
		ENGLISH_LOOKUP.put("thank", "san1 ke4");
		ENGLISH_LOOKUP.put("you", "you1");
		ENGLISH_LOOKUP.put("love", "la1 fu4");
		ENGLISH_LOOKUP.put("cool", "ku3 er4");
		ENGLISH_LOOKUP.put("good", "gu3 de5");
	}

	// 字典路径
	private static final String CHAR_DICT_PATH = "/tts/chinese-char-pinyin.txt";
	private static final String WORD_DICT_PATH = "/tts/chinese-word-pinyin.txt";

	private static volatile Map<String, String> CHAR_DICT;
	private static volatile Map<String, String> WORD_DICT;

	// 会话级 heteronym 覆盖（线程安全 copy-on-write）
	private final Map<Character, String> heteronyms;

	private static volatile ChineseG2P INSTANCE;

	/**
	 * 获取默认实例（单例，延迟初始化，无会话级覆盖）。
	 */
	public static ChineseG2P getDefault() {
		ChineseG2P g = INSTANCE;
		if (g == null) {
			synchronized (ChineseG2P.class) {
				g = INSTANCE;
				if (g == null) {
					g = new ChineseG2P(Collections.emptyMap());
					INSTANCE = g;
				}
			}
		}
		return g;
	}

	/**
	 * 构造带 heteronym 覆盖的 G2P 实例。
	 *
	 * @param heteronyms 字符 → 强制拼音映射（如 行 → hang2）
	 */
	public ChineseG2P(Map<Character, String> heteronyms) {
		this.heteronyms = (heteronyms == null || heteronyms.isEmpty())
			? Collections.emptyMap()
			: new HashMap<>(heteronyms);
	}

	/**
	 * 返回带 heteronym 覆盖的新实例（链式友好）。
	 *
	 * <pre>{@code
	 * G2P g2p = ChineseG2P.getDefault().withHeteronym('行', "hang2");
	 * }</pre>
	 *
	 * @param ch    多音字
	 * @param pinyin 强制拼音（带声调数字）
	 * @return 新的 G2P 实例
	 */
	public ChineseG2P withHeteronym(char ch, String pinyin) {
		Map<Character, String> next = new HashMap<>(heteronyms);
		next.put(ch, pinyin);
		return new ChineseG2P(next);
	}

	/**
	 * 拼音 → 注音符号。公开以便其它 G2P 实现复用。
	 *
	 * <p>支持三种声调标记：</p>
	 * <ul>
	 *   <li>数字声调：1-5（5=轻声），如 "ni3"（字典文件 / 公开 API）</li>
	 *   <li>上标数字：¹²³⁴⁵（Unicode 00B9/00B2/00B3/2074/2075），如 "ni³"（normalizer 输出）</li>
	 * </ul>
	 *
	 * @param pinyin 拼音（带声调标记）
	 * @return 注音符号字符串（如 "ni3" / "ni³" → "ㄋㄧ"）
	 */
	public static String pinyinToBopomofo(String pinyin) {
		if (pinyin == null || pinyin.isEmpty()) return "";
		// 同时剥离数字声调 1-5 和上标数字 ¹²³⁴⁵
		String base = pinyin
			.replaceAll("[1-5]", "")
			.replaceAll("[\u00B9\u00B2\u00B3\u2074\u2075]", "");
		if (base.isEmpty()) return "";

		// 声母
		String initial = "";
		String rest = base;
		for (String ini : INITIAL_KEYS) {
			if (base.startsWith(ini)) {
				initial = INITIALS.get(ini);
				rest = base.substring(ini.length());
				break;
			}
		}
		// y/w 特殊处理（y → ㄧ 但 yu → ㄩ；w → ㄨ）
		if (initial.isEmpty() && base.length() > 1) {
			char head = base.charAt(0);
			if (head == 'y') {
				if (base.length() > 2 && base.charAt(1) == 'u') {
					initial = "ㄩ";
					rest = base.substring(2);
				} else {
					initial = "ㄧ";
					rest = base.substring(1);
				}
			} else if (head == 'w') {
				initial = "ㄨ";
				rest = base.substring(1);
			}
		}

		// 韵母
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

		// 1. 文本归一化（数字 / 金额 / 日期 / 时间 / 百分号 / 电话）
		String normalized = ChineseTextNormalizer.normalize(text);

		// 2. Heteronym 行内标记：$行=hang2$ → 替换为 bopomofo
		StringBuilder out = new StringBuilder();
		int n = normalized.length();
		int i = 0;
		while (i < n) {
			char c = normalized.charAt(i);
			if (c == '$') {
				int end = findHeteronymEnd(normalized, i + 1);
				if (end > 0) {
					String inner = normalized.substring(i + 1, end);
					int eq = inner.indexOf('=');
					if (eq > 0) {
						String pinyin = inner.substring(eq + 1);
						String bpmf = pinyinToBopomofo(pinyin);
						if (!bpmf.isEmpty()) {
							appendWithSpace(out, bpmf);
							i = end + 1;
							continue;
						}
					}
				}
			}
			out.append(c);
			i++;
		}
		String stage = out.toString();

		// 3. 主流分：CJK 段 → 查字典；非 CJK → 英文 / 标点 / 空白
		Map<String, String> charDict = charDict();
		Map<String, String> wordDict = wordDict();
		StringBuilder sb = new StringBuilder();
		int len = stage.length();
		int p = 0;
		while (p < len) {
			char c = stage.charAt(p);

			// 3.1 heteronym 字符优先级：会话级覆盖
			String force = heteronyms.get(c);
			if (force != null) {
				String bpmf = pinyinToBopomofo(force);
				appendWithSpace(sb, bpmf);
				p++;
				continue;
			}

			if (isChinese(c)) {
				// 3.2 词组（最长 4 字）
				boolean matchedWord = false;
				for (int w = 4; w >= 2; w--) {
					if (p + w <= len) {
						String word = stage.substring(p, p + w);
						String pinyinSeq = wordDict.get(word);
						if (pinyinSeq != null) {
							appendPinyinSequence(sb, pinyinSeq);
							p += w;
							matchedWord = true;
							break;
						}
					}
				}
				if (matchedWord) continue;
				// 3.3 单字
				String pinyin = lookupCharPinyin(c, charDict);
				if (pinyin != null) {
					String bpmf = pinyinToBopomofo(pinyin);
					appendWithSpace(sb, bpmf);
				}
				p++;
			} else if (isEnglishLetter(c)) {
				// 3.4 英文段：连续字母合并查表或逐字母
				int start = p;
				while (p < len && isEnglishLetter(stage.charAt(p))) p++;
				String word = stage.substring(start, p);
				String spoken = englishToPinyin(word);
				appendWithSpace(sb, spoken);
			} else if (isPunctuation(c)) {
				sb.append(c);
				p++;
			} else if (Character.isWhitespace(c)) {
				if (!sb.isEmpty() && sb.charAt(sb.length() - 1) != ' ') sb.append(' ');
				p++;
			} else {
				p++;
			}
		}
		return sb.toString().trim();
	}

	private static int findHeteronymEnd(String s, int start) {
		int n = s.length();
		for (int i = start; i < n; i++) {
			if (s.charAt(i) == '$') return i;
		}
		return -1;
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
		int slash = pinyinSeq.indexOf('/');
		return slash < 0 ? pinyinSeq : pinyinSeq.substring(0, slash);
	}

	/**
	 * 把英文单词/缩写转成 pinyin 串。
	 * <ol>
	 *   <li>命中 ENGLISH_LOOKUP 表 → 返回该中文读音</li>
	 *   <li>否则逐字母（A → ei4，B → bi4 ...）</li>
	 * </ol>
	 */
	private static String englishToPinyin(String word) {
		String looked = ENGLISH_LOOKUP.get(word);
		if (looked != null) return looked;

		// 全部字母单读
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < word.length(); i++) {
			char c = word.charAt(i);
			String letter = ENGLISH_LETTERS.get(c);
			if (letter != null) {
				if (sb.length() > 0) sb.append(' ');
				sb.append(letter);
			}
		}
		return sb.toString();
	}

	// 字母 → 拼音（英文字母表，按字母读音近似）
	private static final Map<Character, String> ENGLISH_LETTERS = new HashMap<>();
	static {
		ENGLISH_LETTERS.put('A', "ei4"); ENGLISH_LETTERS.put('B', "bi4");
		ENGLISH_LETTERS.put('C', "xi4");  ENGLISH_LETTERS.put('D', "di4");
		ENGLISH_LETTERS.put('E', "yi4");  ENGLISH_LETTERS.put('F', "ai2 fu4");
		ENGLISH_LETTERS.put('G', "ji4");  ENGLISH_LETTERS.put('H', "ai2 chi4");
		ENGLISH_LETTERS.put('I', "ai4");  ENGLISH_LETTERS.put('J', "jie2");
		ENGLISH_LETTERS.put('K', "kei4"); ENGLISH_LETTERS.put('L', "ai2 er4");
		ENGLISH_LETTERS.put('M', "ai2 mu4"); ENGLISH_LETTERS.put('N', "en1");
		ENGLISH_LETTERS.put('O', "ou1"); ENGLISH_LETTERS.put('P', "pi4");
		ENGLISH_LETTERS.put('Q', "kiu4"); ENGLISH_LETTERS.put('R', "a4 er4");
		ENGLISH_LETTERS.put('S', "ai2 si4"); ENGLISH_LETTERS.put('T', "ti4");
		ENGLISH_LETTERS.put('U', "you1"); ENGLISH_LETTERS.put('V', "wi4");
		ENGLISH_LETTERS.put('W', "da4 bu4 liu4"); ENGLISH_LETTERS.put('X', "ai2 ke4 si4");
		ENGLISH_LETTERS.put('Y', "wai4"); ENGLISH_LETTERS.put('Z', "ze2");
		// 小写字母（大写小写读法一致）
		ENGLISH_LETTERS.put('a', "ei4"); ENGLISH_LETTERS.put('b', "bi4");
		ENGLISH_LETTERS.put('c', "xi4"); ENGLISH_LETTERS.put('d', "di4");
		ENGLISH_LETTERS.put('e', "yi4"); ENGLISH_LETTERS.put('f', "ai2 fu4");
		ENGLISH_LETTERS.put('g', "ji4"); ENGLISH_LETTERS.put('h', "ai2 chi4");
		ENGLISH_LETTERS.put('i', "ai4"); ENGLISH_LETTERS.put('j', "jie2");
		ENGLISH_LETTERS.put('k', "kei4"); ENGLISH_LETTERS.put('l', "ai2 er4");
		ENGLISH_LETTERS.put('m', "ai2 mu4"); ENGLISH_LETTERS.put('n', "en1");
		ENGLISH_LETTERS.put('o', "ou1"); ENGLISH_LETTERS.put('p', "pi4");
		ENGLISH_LETTERS.put('q', "kiu4"); ENGLISH_LETTERS.put('r', "a4 er4");
		ENGLISH_LETTERS.put('s', "ai2 si4"); ENGLISH_LETTERS.put('t', "ti4");
		ENGLISH_LETTERS.put('u', "you1"); ENGLISH_LETTERS.put('v', "wi4");
		ENGLISH_LETTERS.put('w', "da4 bu4 liu4"); ENGLISH_LETTERS.put('x', "ai2 ke4 si4");
		ENGLISH_LETTERS.put('y', "wai4"); ENGLISH_LETTERS.put('z', "ze2");
	}

	private static void appendWithSpace(StringBuilder sb, String s) {
		if (s == null || s.isEmpty()) return;
		if (!sb.isEmpty() && sb.charAt(sb.length() - 1) != ' ') sb.append(' ');
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
	// 字典加载（首次访问懒加载，线程安全）
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
		if (in == null) return m;
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
				if (!key.isEmpty() && !value.isEmpty()) m.put(key, value);
			}
		} catch (Exception ignored) {
		}
		return m;
	}
}
