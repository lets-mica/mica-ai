package net.dreamlu.mica.ai.tts.g2p;

import com.github.houbb.pinyin.constant.enums.PinyinStyleEnum;
import com.github.houbb.pinyin.util.PinyinHelper;

/**
 * 基于 <a href="https://github.com/houbb/pinyin">houbb/pinyin</a> 的高质量中文 G2P 实现（默认实现）。
 *
 * <p>特性：
 * <ul>
 *     <li>支持多音字智能消歧（基于分词）</li>
 *     <li>支持中文分词（"重庆火锅" → "chóng qìng huǒ guō"）</li>
 *     <li>支持繁简体</li>
 *     <li>支持自定义拼音词典</li>
 * </ul>
 *
 * <p>输出格式：拼音+数字声调（"wo3 ai4 zhong1 wen2"），再经由 {@link BopomofoConverter#convert(String)}
 * 转换为 Kokoro 所需的注音符号。
 *
 * @author L.cm
 */
public class HoubbPinyinG2P implements G2P {
	private final char separator = ' ';

	@Override
	public String convert(String text) {
		if (text == null || text.isEmpty()) return "";
		StringBuilder sb = new StringBuilder();
		StringBuilder chineseBuf = new StringBuilder();
		for (int i = 0; i < text.length(); i++) {
			char c = text.charAt(i);
			if (isChinese(c)) {
				chineseBuf.append(c);
			} else {
				flushChinese(sb, chineseBuf);
				if (Character.isDigit(c)) {
					appendWithSpace(sb, BopomofoConverter.convert(digitToPinyin(c)));
				} else if (isEnglishLetter(c)) {
					sb.append(c);
				} else if (isPunctuation(c)) {
					// 标点两侧补空格：避免 vocab.filter 丢全角标点后音素粘连
					if (!sb.isEmpty() && sb.charAt(sb.length() - 1) != ' ') {
						sb.append(separator);
					}
				} else if (Character.isWhitespace(c)) {
					if (!sb.isEmpty() && sb.charAt(sb.length() - 1) != ' ') {
						sb.append(separator);
					}
				}
			}
		}
		flushChinese(sb, chineseBuf);
		return sb.toString().trim();
	}

	private void flushChinese(StringBuilder sb, StringBuilder chineseBuf) {
		if (chineseBuf.isEmpty()) {
			return;
		}
		String pinyinStr = PinyinHelper.toPinyin(chineseBuf.toString(), PinyinStyleEnum.NUM_LAST);
		String[] syllables = pinyinStr.split("\\s+");
		for (String syllable : syllables) {
			if (syllable.isEmpty()) continue;
			// houbb/pinyin 输出 "zhong1" 这种 拼音+数字声调 格式
			// Kokoro 训练数据中每个 Bopomofo 音节后必须跟随一个声调数字 token (U+0031-U+0035)
			int tone = 0;
			String pinyinBase = syllable;
			char last = syllable.charAt(syllable.length() - 1);
			if (last >= '1' && last <= '5') {
				tone = last - '0';
				pinyinBase = syllable.substring(0, syllable.length() - 1);
			}
			String bpmf = BopomofoConverter.convert(pinyinBase);
			if (bpmf.isEmpty()) continue;
			appendWithSpace(sb, bpmf);
			if (tone > 0) {
				sb.append((char) ('0' + tone));
			}
		}
		chineseBuf.setLength(0);
	}

	private static String digitToPinyin(char c) {
		return switch (c) {
			case '0' -> "ling2";
			case '1' -> "yi1";
			case '2' -> "er4";
			case '3' -> "san1";
			case '4' -> "si4";
			case '5' -> "wu3";
			case '6' -> "liu4";
			case '7' -> "qi1";
			case '8' -> "ba1";
			case '9' -> "jiu3";
			default -> "";
		};
	}

	private void appendWithSpace(StringBuilder sb, String s) {
		if (s == null || s.isEmpty()) return;
		if (!sb.isEmpty() && sb.charAt(sb.length() - 1) != ' ') {
			sb.append(separator);
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
