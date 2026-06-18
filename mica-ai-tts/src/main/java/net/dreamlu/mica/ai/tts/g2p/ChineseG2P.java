package net.dreamlu.mica.ai.tts.g2p;

import java.util.*;

/**
 * 简化版中文 G2P 转换器（fallback 实现）。
 * <p>使用内置汉字-拼音字典 + 拼音→注音符号映射表实现。
 * <p>覆盖约 80 个常用汉字，陌生字符会按原样保留（vocab 会过滤）。
 *
 * <p>生产环境推荐使用 {@code net.dreamlu.mica.ai.tts.g2p.HoubbPinyinG2P}
 * （基于 houbb/pinyin，支持多音字、中文分词、繁简体）。
 *
 * @author L.cm
 */
public final class ChineseG2P implements G2P {

	// 声母 → 注音符号映射
	private static final Map<String, String> INITIALS = new LinkedHashMap<>();
	static {
		INITIALS.put("zh", "ㄓ"); INITIALS.put("ch", "ㄔ"); INITIALS.put("sh", "ㄕ");
		INITIALS.put("b", "ㄅ"); INITIALS.put("p", "ㄆ"); INITIALS.put("m", "ㄇ"); INITIALS.put("f", "ㄈ");
		INITIALS.put("d", "ㄉ"); INITIALS.put("t", "ㄊ"); INITIALS.put("n", "ㄋ"); INITIALS.put("l", "ㄌ");
		INITIALS.put("g", "ㄍ"); INITIALS.put("k", "ㄎ"); INITIALS.put("h", "ㄏ");
		INITIALS.put("j", "ㄐ"); INITIALS.put("q", "ㄑ"); INITIALS.put("x", "ㄒ");
		INITIALS.put("r", "ㄖ"); INITIALS.put("z", "ㄗ"); INITIALS.put("c", "ㄘ"); INITIALS.put("s", "ㄙ");
		INITIALS.put("y", ""); INITIALS.put("w", "");
	}

	// 韵母 → 注音符号映射
	private static final Map<String, String> FINALS = new LinkedHashMap<>();
	static {
		FINALS.put("iang", "ㄧㄤ"); FINALS.put("iong", "ㄩㄥ");
		FINALS.put("uang", "ㄨㄤ");
		FINALS.put("uai", "ㄨㄞ"); FINALS.put("uan", "ㄨㄢ");
		FINALS.put("ang", "ㄤ"); FINALS.put("eng", "ㄥ"); FINALS.put("ing", "ㄧㄥ");
		FINALS.put("ong", "ㄨㄥ"); FINALS.put("ian", "ㄧㄢ"); FINALS.put("iao", "ㄧㄠ");
		FINALS.put("an", "ㄢ"); FINALS.put("en", "ㄣ"); FINALS.put("in", "ㄧㄣ");
		FINALS.put("un", "ㄨㄣ"); FINALS.put("ai", "ㄞ"); FINALS.put("ei", "ㄟ");
		FINALS.put("ao", "ㄠ"); FINALS.put("ou", "ㄡ");
		FINALS.put("ia", "ㄧㄚ"); FINALS.put("ie", "ㄧㄝ");
		FINALS.put("iu", "ㄧㄡ"); FINALS.put("ua", "ㄨㄚ");
		FINALS.put("uo", "ㄨㄛ"); FINALS.put("ui", "ㄨㄟ");
		FINALS.put("ve", "ㄩㄝ"); FINALS.put("van", "ㄩㄢ"); FINALS.put("vn", "ㄩㄣ");
		FINALS.put("a", "ㄚ"); FINALS.put("o", "ㄛ"); FINALS.put("e", "ㄜ");
		FINALS.put("i", "ㄧ"); FINALS.put("u", "ㄨ"); FINALS.put("v", "ㄩ");
		FINALS.put("er", "ㄦ");
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

	// 常见汉字 → 拼音（仅作为最简 fallback）
	private static final Map<Character, String> CHAR_FALLBACK = new HashMap<>();
	static {
		CHAR_FALLBACK.put('你', "ni3"); CHAR_FALLBACK.put('好', "hao3");
		CHAR_FALLBACK.put('我', "wo3"); CHAR_FALLBACK.put('是', "shi4");
		CHAR_FALLBACK.put('的', "de5"); CHAR_FALLBACK.put('了', "le5");
		CHAR_FALLBACK.put('在', "zai4"); CHAR_FALLBACK.put('有', "you3");
		CHAR_FALLBACK.put('人', "ren2"); CHAR_FALLBACK.put('这', "zhe4");
		CHAR_FALLBACK.put('中', "zhong1"); CHAR_FALLBACK.put('大', "da4");
		CHAR_FALLBACK.put('来', "lai2"); CHAR_FALLBACK.put('上', "shang4");
		CHAR_FALLBACK.put('国', "guo2"); CHAR_FALLBACK.put('个', "ge4");
		CHAR_FALLBACK.put('到', "dao4"); CHAR_FALLBACK.put('说', "shuo1");
		CHAR_FALLBACK.put('们', "men5"); CHAR_FALLBACK.put('为', "wei2");
		CHAR_FALLBACK.put('子', "zi3"); CHAR_FALLBACK.put('和', "he2");
		CHAR_FALLBACK.put('不', "bu4"); CHAR_FALLBACK.put('地', "di4");
		CHAR_FALLBACK.put('出', "chu1"); CHAR_FALLBACK.put('时', "shi2");
		CHAR_FALLBACK.put('年', "nian2"); CHAR_FALLBACK.put('得', "de2");
		CHAR_FALLBACK.put('就', "jiu4"); CHAR_FALLBACK.put('那', "na4");
		CHAR_FALLBACK.put('要', "yao4"); CHAR_FALLBACK.put('下', "xia4");
		CHAR_FALLBACK.put('以', "yi3"); CHAR_FALLBACK.put('生', "sheng1");
		CHAR_FALLBACK.put('会', "hui4"); CHAR_FALLBACK.put('自', "zi4");
		CHAR_FALLBACK.put('着', "zhe5"); CHAR_FALLBACK.put('去', "qu4");
		CHAR_FALLBACK.put('之', "zhi1"); CHAR_FALLBACK.put('过', "guo4");
		CHAR_FALLBACK.put('家', "jia1"); CHAR_FALLBACK.put('学', "xue2");
		CHAR_FALLBACK.put('对', "dui4"); CHAR_FALLBACK.put('可', "ke3");
		CHAR_FALLBACK.put('她', "ta1"); CHAR_FALLBACK.put('他', "ta1");
		CHAR_FALLBACK.put('里', "li3"); CHAR_FALLBACK.put('后', "hou4");
		CHAR_FALLBACK.put('小', "xiao3"); CHAR_FALLBACK.put('么', "me5");
		CHAR_FALLBACK.put('心', "xin1"); CHAR_FALLBACK.put('多', "duo1");
		CHAR_FALLBACK.put('天', "tian1"); CHAR_FALLBACK.put('而', "er2");
		CHAR_FALLBACK.put('能', "neng2");
		CHAR_FALLBACK.put('看', "kan4"); CHAR_FALLBACK.put('当', "dang1");
		CHAR_FALLBACK.put('没', "mei2"); CHAR_FALLBACK.put('日', "ri4");
		CHAR_FALLBACK.put('于', "yu2"); CHAR_FALLBACK.put('起', "qi3");
		CHAR_FALLBACK.put('还', "hai2"); CHAR_FALLBACK.put('发', "fa1");
		CHAR_FALLBACK.put('成', "cheng2"); CHAR_FALLBACK.put('事', "shi4");
		CHAR_FALLBACK.put('只', "zhi3"); CHAR_FALLBACK.put('作', "zuo4");
		CHAR_FALLBACK.put('用', "yong4"); CHAR_FALLBACK.put('想', "xiang3");
		CHAR_FALLBACK.put('把', "ba3");
		CHAR_FALLBACK.put('十', "shi2"); CHAR_FALLBACK.put('月', "yue4");
		CHAR_FALLBACK.put('千', "qian1");
		CHAR_FALLBACK.put('行', "xing2");
		CHAR_FALLBACK.put('始', "shi3");
		CHAR_FALLBACK.put('足', "zu2");
	}

	private static volatile ChineseG2P INSTANCE;

	/**
	 * 获取默认实例（单例）。
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
	 * 将拼音（如 "ni3", "hao3"）转换为注音符号。
	 * <p>该方法公开，方便其它 G2P 实现复用。
	 *
	 * @param pinyin 拼音（带数字声调）
	 * @return 注音符号字符串
	 */
	public static String pinyinToBopomofo(String pinyin) {
		if (pinyin == null || pinyin.isEmpty()) return "";

		// 去除声调数字
		String base = pinyin.replaceAll("[1-5]", "");
		if (base.isEmpty()) return "";

		// 提取声母
		String initial = "";
		String finals = base;
		for (String ini : INITIALS.keySet()) {
			if (base.startsWith(ini)) {
				initial = INITIALS.get(ini);
				finals = base.substring(ini.length());
				break;
			}
		}

		// 特殊处理：y/w 开头的音节
		if (base.startsWith("y") || base.startsWith("w")) {
			initial = "";
			finals = base.substring(1);
		}

		// 查找韵母
		String finalBpmf = FINALS.getOrDefault(finals, "");

		return initial + finalBpmf;
	}

	@Override
	public String convert(String text) {
		if (text == null || text.isEmpty()) return "";
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < text.length(); i++) {
			char c = text.charAt(i);
			if (isChinese(c)) {
				String pinyin = CHAR_FALLBACK.get(c);
				if (pinyin != null) {
					String bpmf = pinyinToBopomofo(pinyin);
					appendWithSpace(sb, bpmf);
				}
				// 未知汉字丢弃（vocab 过滤），避免污染音素序列
			} else if (Character.isDigit(c)) {
				String pinyin = NUMBER_MAP.get(c);
				if (pinyin != null) {
					String bpmf = pinyinToBopomofo(pinyin);
					appendWithSpace(sb, bpmf);
				}
			} else if (isEnglishLetter(c)) {
				sb.append(c);
			} else if (isPunctuation(c)) {
				sb.append(c);
			} else if (Character.isWhitespace(c)) {
				// 直接追加空格，不经过 appendWithSpace（空串会被跳过）
				if (!sb.isEmpty() && sb.charAt(sb.length() - 1) != ' ') {
					sb.append(' ');
				}
			}
		}
		return sb.toString().trim();
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
		return ",.!?;:，。！？；：、（）()\"\"—…".indexOf(c) >= 0;
	}
}
