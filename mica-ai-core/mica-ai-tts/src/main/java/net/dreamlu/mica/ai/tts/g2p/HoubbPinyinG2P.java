package net.dreamlu.mica.ai.tts.g2p;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;

/**
 * 基于 <a href="https://github.com/houbb/pinyin">houbb/pinyin</a> 的高质量中文 G2P 实现。
 *
 * <p>特性：
 * <ul>
 *     <li>支持多音字智能消歧（基于分词）</li>
 *     <li>支持中文分词（"重庆火锅" → "chóng qìng huǒ guō"）</li>
 *     <li>支持繁简体</li>
 *     <li>支持自定义拼音词典</li>
 * </ul>
 *
 * <p>输出格式：拼音+数字声调（"wo3 ai4 zhong1 wen2"），再经由 {@link ChineseG2P#pinyinToBopomofo(String)}
 * 转换为 Kokoro 所需的注音符号。
 *
 * <p><b>使用方式</b>：在项目中显式添加依赖：
 * <pre>{@code
 * <dependency>
 *     <groupId>com.github.houbb</groupId>
 *     <artifactId>pinyin</artifactId>
 *     <version>0.4.0</version>
 * </dependency>
 * }</pre>
 *
 * <p><b>注意</b>：本类使用反射调用 houbb 库，避免 mica-ai-tts 对其产生强依赖。
 * 如果未引入 houbb 库，调用 {@link #convert(String)} 时会抛 {@link IllegalStateException}。
 *
 * @author L.cm
 */
public class HoubbPinyinG2P implements G2P {

	private final Method toPinyinMethod;
	private final boolean initialized;
	private final String separator = " ";

	/**
	 * 创建默认实例（反射加载 houbb/pinyin）。
	 *
	 * @throws IllegalStateException 未找到 houbb/pinyin 依赖
	 */
	public HoubbPinyinG2P() {
		Method m = null;
		boolean ok = false;
		try {
			// 反射加载 com.github.houbb.pinyin.PinyinHelper
			// API: PinyinHelper.toPinyin(String, PinyinStyleEnum, String)
			// 反射：避免 mica-ai-tts 对 houbb 库的强依赖
			Class<?> styleEnum = Class.forName("com.github.houbb.pinyin.constant.enums.PinyinStyleEnum");
			Object numLast = null;
			for (Object constant : styleEnum.getEnumConstants()) {
				if ("NUM_LAST".equals(((Enum<?>) constant).name())) {
					numLast = constant;
					break;
				}
			}
			if (numLast == null) {
				throw new IllegalStateException("PinyinStyleEnum.NUM_LAST not found in houbb/pinyin");
			}
			// 实际类位于 com.github.houbb.pinyin.util.PinyinHelper
			Class<?> helper = Class.forName("com.github.houbb.pinyin.util.PinyinHelper");
			// 找到签名 (String, PinyinStyleEnum) 的 toPinyin 方法
			for (Method method : helper.getMethods()) {
				if ("toPinyin".equals(method.getName())
					&& method.getParameterCount() == 2
					&& method.getParameterTypes()[0] == String.class
					&& method.getParameterTypes()[1] == styleEnum) {
					m = method;
					break;
				}
			}
			if (m == null) {
				throw new IllegalStateException("PinyinHelper.toPinyin(String, PinyinStyleEnum) not found");
			}
			ok = true;
		} catch (ClassNotFoundException e) {
			// houbb/pinyin 未引入，保持非初始化状态
		}
		this.toPinyinMethod = m;
		this.initialized = ok;
	}

	/**
	 * 检查 houbb/pinyin 库是否可用。
	 */
	public boolean isAvailable() {
		return initialized;
	}

	@Override
	public String convert(String text) {
		if (text == null || text.isEmpty()) return "";
		if (!initialized) {
			throw new IllegalStateException(
				"houbb/pinyin 库未找到，请添加依赖：\n"
					+ "<dependency>\n"
					+ "    <groupId>com.github.houbb</groupId>\n"
					+ "    <artifactId>pinyin</artifactId>\n"
					+ "    <version>0.4.0</version>\n"
					+ "</dependency>");
		}
		StringBuilder sb = new StringBuilder();
		// 按字符处理：中文 → 拼音转换；英文/数字/标点/空白 → 保留
		StringBuilder chineseBuf = new StringBuilder();
		for (int i = 0; i < text.length(); i++) {
			char c = text.charAt(i);
			if (isChinese(c)) {
				chineseBuf.append(c);
			} else {
				flushChinese(sb, chineseBuf);
				if (Character.isDigit(c)) {
					appendWithSpace(sb, ChineseG2P.pinyinToBopomofo(digitToPinyin(c)));
				} else if (isEnglishLetter(c)) {
					sb.append(c);
				} else if (isPunctuation(c)) {
					sb.append(c);
				} else if (Character.isWhitespace(c)) {
					// 直接追加空格，不经过 appendWithSpace（空串会被跳过）
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
		String pinyinStr = toPinyin(chineseBuf.toString());
		// 拼音格式: "wo3 ai4 zhong1 wen2"
		String[] syllables = pinyinStr.split("\\s+");
		for (String syllable : syllables) {
			if (!syllable.isEmpty()) {
				String bpmf = ChineseG2P.pinyinToBopomofo(syllable);
				appendWithSpace(sb, bpmf);
			}
		}
		chineseBuf.setLength(0);
	}

	private String toPinyin(String chinese) {
		try {
			Object result = toPinyinMethod.invoke(null, chinese, getNumLastStyle());
			return result == null ? "" : result.toString();
		} catch (IllegalAccessException | InvocationTargetException e) {
			throw new IllegalStateException("Failed to call PinyinHelper.toPinyin: " + e.getMessage(), e);
		}
	}

	private Object numLastStyleCache;

	private Object getNumLastStyle() {
		if (numLastStyleCache != null) return numLastStyleCache;
		try {
			Class<?> styleEnum = Class.forName("com.github.houbb.pinyin.constant.enums.PinyinStyleEnum");
			for (Object constant : styleEnum.getEnumConstants()) {
				if ("NUM_LAST".equals(((Enum<?>) constant).name())) {
					numLastStyleCache = constant;
					return constant;
				}
			}
		} catch (ClassNotFoundException ignored) {
		}
		throw new IllegalStateException("PinyinStyleEnum.NUM_LAST not found");
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
