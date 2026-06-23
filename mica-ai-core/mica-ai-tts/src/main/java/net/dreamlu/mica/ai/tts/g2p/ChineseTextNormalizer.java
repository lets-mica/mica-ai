package net.dreamlu.mica.ai.tts.g2p;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 简体中文文本归一化器（TTS 预处理）。
 *
 * <p>职责：在 G2P 之前把非汉字文本转换成汉字读法，避免 Kokoro vocab 过滤掉标点/数字。</p>
 *
 * <p>处理范围：</p>
 * <ol>
 *   <li><b>数字读法</b>：阿拉伯数字 → 中文读音（支持负数、小数、百分号）</li>
 *   <li><b>金额</b>：¥/$ 开头 + 数字 → "X 元 / 美元"</li>
 *   <li><b>年份</b>：4 位数字 + 年 → "二零二六年"</li>
 *   <li><b>日期</b>：YYYY-MM-DD / YYYY年M月D日 → "二零二六年六月二十四日"</li>
 *   <li><b>时间</b>：HH:MM / HH 点 MM 分 → "十二点三十四分"</li>
 *   <li><b>百分号</b>：50% → "百分之五十"</li>
 *   <li><b>电话号码</b>：连续 7+ 位数字 → 逐位读</li>
 *   <li><b>特殊符号</b>：全角空格、HTML 实体、控制字符清理</li>
 * </ol>
 *
 * <p>所有归一化都是<b>纯正则 + 查表</b>，零依赖；可独立注入到任意 G2P。</p>
 *
 * @author L.cm
 */
public final class ChineseTextNormalizer {

	private ChineseTextNormalizer() {}

	// 上标数字（声调标记，避开阿拉伯数字 1-5 与 normalizer 二次翻译冲突）
	private static final String T1 = "\u00B9";
	private static final String T2 = "\u00B2";
	private static final String T3 = "\u00B3";
	private static final String T4 = "\u2074";
	private static final String T5 = "\u2075";

	// 数字 0-9 读音（零到九）。tone 用上标数字 ¹²³⁴⁵ 表示
	// （避开阿拉伯数字 1-5，否则 normalize 输出的 bai3 又会被 normalizeNumber 当作数字二次翻译；
	//   也避开 a-e 字母，否则会误删 pinyin 里的 a/e 字符）
	private static final String[] DIGIT_PINYIN = {
		"ling" + T5, "yi" + T1, "er" + T4, "san" + T1, "si" + T4,
		"wu" + T3, "liu" + T4, "qi" + T1, "ba" + T1, "jiu" + T3
	};

	// 中文数字单位（同样用上标数字声调）
	private static final String[] UNIT_LOW = {"", "shi" + T2, "bai" + T3, "qian" + T1};       // 个十百千
	private static final String[] UNIT_HIGH = {"", "wan" + T4, "yi" + T4};                     // 万亿
	private static final String ZERO = "ling" + T5;

	// 金额单位（元 / 块）
	private static final String YUAN = "yuan" + T2;
	private static final String Kuai = "kuai" + T4;
	private static final String JIAO = "jiao" + T1;
	private static final String FEN = "fen" + T1;
	private static final String DOLLAR = "mei" + T3 + " yuan" + T2;

	// 年月日时分秒
	private static final String NIAN = "nian" + T2;
	private static final String YUE = "yue" + T4;
	private static final String RI = "ri" + T4;
	private static final String DIAN = "dian" + T3;
	private static final String FEN2 = "fen" + T1;
	private static final String MIAO = "miao" + T3;

	// 百分号 / 千分号
	private static final String BAI_FEN_ZHI = "bai" + T3 + " fen" + T1 + " zhi" + T1;
	private static final String QIAN_FEN_ZHI = "qian" + T1 + " fen" + T1 + " zhi" + T1;

	// 复合格式
	private static final Pattern RE_DATE_ISO = Pattern.compile("(\\d{4})[-/.](\\d{1,2})[-/.](\\d{1,2})");
	private static final Pattern RE_YEAR_MONTH = Pattern.compile("(\\d{4})\\s*年\\s*(?:(\\d{1,2})\\s*月\\s*(?:(\\d{1,2})\\s*日?)?)?");
	private static final Pattern RE_TIME = Pattern.compile("(\\d{1,2}):(\\d{1,2})(?::(\\d{1,2}))?");
	private static final Pattern RE_TIME_CN = Pattern.compile("(\\d{1,2})\\s*[点时]\\s*(\\d{1,2})?\\s*分?(?:\\s*(\\d{1,2})\\s*秒?)?");
	private static final Pattern RE_PERCENT = Pattern.compile("(-?\\d+(?:\\.\\d+)?)\\s*%");
	private static final Pattern RE_CURRENCY_CN = Pattern.compile("[¥￥]\\s*(-?\\d+(?:,\\d{3})*(?:\\.\\d+)?)");
	private static final Pattern RE_CURRENCY_EN = Pattern.compile("[$]\\s*(-?\\d+(?:,\\d{3})*(?:\\.\\d+)?)");
	private static final Pattern RE_NUMBER = Pattern.compile("(?<![\\d.])(-?\\d{1,16}(?:,\\d{3})*(?:\\.\\d+)?)(?![\\d.])");
	private static final Pattern RE_PHONE = Pattern.compile("(?<![\\d-])(\\d{7,15})(?![\\d-])");
	private static final Pattern RE_THAW = Pattern.compile("[\\u00A0\\u2003-\\u200B\\u2028\\u2029\\uFEFF]");
	private static final Pattern RE_MULTI_SPACE = Pattern.compile("[ \\t]{2,}");

	/**
	 * 把待合成文本转换为 G2P 友好的形式（非汉字读法替换为汉字读法）。
	 *
	 * @param text 原始文本
	 * @return 归一化后文本（仍是汉字/中文数字/中英混合，不含 ¥$% 等符号）
	 */
	public static String normalize(String text) {
		if (text == null || text.isEmpty()) return "";
		String s = text;

		// 1. 清理不可见字符 / 多余空白
		s = RE_THAW.matcher(s).replaceAll(" ");
		s = RE_MULTI_SPACE.matcher(s).replaceAll(" ");

		// 2. 日期 ISO → 汉字读法（必须在数字归一化之前！）
		s = normalizeIsoDate(s);

		// 3. 中文年月（YYYY年M月D日）
		s = normalizeYearMonth(s);

		// 4. 时间 (HH:MM:SS)
		s = normalizeTime(s);

		// 5. 百分号 / 千分号
		s = normalizePercent(s);

		// 6. 金额 ¥/$ 开头
		s = normalizeCurrency(s);

		// 7. 电话号码（≥7 位连续数字）→ 逐位
		s = normalizePhone(s);

		// 8. 剩余纯数字（避免重复处理上面已转的）
		s = normalizeNumber(s);

		return s.trim();
	}

	// ====================================================================
	// 各规则实现
	// ====================================================================

	private static String normalizeIsoDate(String s) {
		Matcher m = RE_DATE_ISO.matcher(s);
		StringBuffer sb = new StringBuffer();
		while (m.find()) {
			String year = yearDigitsToChinese(m.group(1));
			String month = monthToChinese(m.group(2));
			String day = dayToChinese(m.group(3));
			m.appendReplacement(sb, Matcher.quoteReplacement(year + NIAN + " " + month + YUE + " " + day + RI));
		}
		m.appendTail(sb);
		return sb.toString();
	}

	private static String normalizeYearMonth(String s) {
		Matcher m = RE_YEAR_MONTH.matcher(s);
		StringBuffer sb = new StringBuffer();
		while (m.find()) {
			StringBuilder out = new StringBuilder();
			out.append(yearDigitsToChinese(m.group(1))).append(" ").append(NIAN);
			if (m.group(2) != null) {
				out.append(' ').append(monthToChinese(m.group(2))).append(' ').append(YUE);
				if (m.group(3) != null) {
					out.append(' ').append(dayToChinese(m.group(3))).append(' ').append(RI);
				}
			}
			m.appendReplacement(sb, Matcher.quoteReplacement(out.toString()));
		}
		m.appendTail(sb);
		return sb.toString();
	}

	private static String normalizeTime(String s) {
		Matcher m = RE_TIME.matcher(s);
		StringBuffer sb = new StringBuffer();
		while (m.find()) {
			int hh = Integer.parseInt(m.group(1));
			int mm = Integer.parseInt(m.group(2));
			StringBuilder out = new StringBuilder();
			out.append(numToChinese(hh, false)).append(" ").append(DIAN).append(" ");
			out.append(numToChinese(mm, false)).append(" ").append(FEN2);
			if (m.group(3) != null) {
				int ss = Integer.parseInt(m.group(3));
				out.append(" ").append(numToChinese(ss, false)).append(" ").append(MIAO);
			}
			m.appendReplacement(sb, Matcher.quoteReplacement(out.toString()));
		}
		m.appendTail(sb);
		return sb.toString();
	}

	private static String normalizePercent(String s) {
		Matcher m = RE_PERCENT.matcher(s);
		StringBuffer sb = new StringBuffer();
		while (m.find()) {
			String num = m.group(1);
			String spoken;
			if (num.contains(".")) {
				// 0.5% → 百分之零点五
				String[] parts = num.split("\\.");
				spoken = BAI_FEN_ZHI + " " + numToChinese(Long.parseLong(parts[0]), false) + " " + DIAN + " " + digitsToChinese(parts[1]);
			} else {
				spoken = BAI_FEN_ZHI + " " + numToChinese(Long.parseLong(num), false);
			}
			m.appendReplacement(sb, Matcher.quoteReplacement(spoken));
		}
		m.appendTail(sb);
		return sb.toString();
	}

	private static String normalizeCurrency(String s) {
		s = replaceCurrency(s, RE_CURRENCY_CN, false);
		s = replaceCurrency(s, RE_CURRENCY_EN, true);
		return s;
	}

	private static String replaceCurrency(String s, Pattern p, boolean isUsd) {
		Matcher m = p.matcher(s);
		StringBuffer sb = new StringBuffer();
		while (m.find()) {
			String raw = m.group(1).replace(",", "");
			double value = Double.parseDouble(raw);
			String spoken = moneyToChinese(value, isUsd);
			m.appendReplacement(sb, Matcher.quoteReplacement(spoken));
		}
		m.appendTail(sb);
		return sb.toString();
	}

	private static String normalizePhone(String s) {
		Matcher m = RE_PHONE.matcher(s);
		StringBuffer sb = new StringBuffer();
		while (m.find()) {
			String digits = m.group(1);
			String spoken = digitsToChinese(digits);
			m.appendReplacement(sb, Matcher.quoteReplacement(spoken));
		}
		m.appendTail(sb);
		return sb.toString();
	}

	private static String normalizeNumber(String s) {
		Matcher m = RE_NUMBER.matcher(s);
		StringBuffer sb = new StringBuffer();
		while (m.find()) {
			String raw = m.group(1).replace(",", "");
			if (raw.startsWith("-")) {
				String spoken = "fu\u2074 " + numToChinese(Long.parseLong(raw.substring(1)), true);
				m.appendReplacement(sb, Matcher.quoteReplacement(spoken));
				continue;
			}
			if (raw.contains(".")) {
				String[] parts = raw.split("\\.");
				String spoken = numToChinese(Long.parseLong(parts[0]), true) + " " + DIAN + " " + digitsToChinese(parts[1]);
				m.appendReplacement(sb, Matcher.quoteReplacement(spoken));
				continue;
			}
			String spoken = numToChinese(Long.parseLong(raw), true);
			m.appendReplacement(sb, Matcher.quoteReplacement(spoken));
		}
		m.appendTail(sb);
		return sb.toString();
	}

	// ====================================================================
	// 中文数字读法（核心）
	// ====================================================================

	/**
	 * 把整数 0..Long.MAX_VALUE 转成中文读音拼音串（空格分隔音节）。
	 *
	 * @param n 待转换整数
	 * @param withUnit true=带单位（一万、亿），适合普通数字；false=逐位（适合时间、月份）
	 */
	public static String numToChinese(long n, boolean withUnit) {
		if (n == 0) return ZERO;
		if (n < 0) return "fu" + T4 + " " + numToChinese(-n, withUnit);
		// 时间/月份/日期场合的 10..19：日常读法为 "十二/十三..."，不带 "yi1 shi2" 中的 "yi"
		if (!withUnit && n >= 10 && n <= 19) {
			return UNIT_LOW[1] + " " + DIGIT_PINYIN[(int) (n - 10)];
		}
		return withUnit ? intToChineseWithUnit(n) : intToChineseDigit(n);
	}

	private static String intToChineseDigit(long n) {
		StringBuilder sb = new StringBuilder();
		String digits = String.valueOf(n);
		for (int i = 0; i < digits.length(); i++) {
			int d = digits.charAt(i) - '0';
			if (i > 0) sb.append(' ');
			sb.append(DIGIT_PINYIN[d]);
		}
		return sb.toString();
	}

	private static String intToChineseWithUnit(long n) {
		if (n == 0) return ZERO;
		StringBuilder sb = new StringBuilder();
		// 处理亿部分
		long yi = n / 100_000_000L;
		long rest = n % 100_000_000L;
		if (yi > 0) {
			sb.append(section(yi)).append(" ").append(UNIT_HIGH[1]).append(' ');
		}
		// 处理万部分
		long wan = rest / 10000L;
		rest = rest % 10000L;
		if (wan > 0) {
			// 万部分独立：50005 → "五万零五"
			StringBuilder wsb = new StringBuilder();
			wsb.append(section4(wan));
			// wan4 前的 0 处理
			if (rest > 0 && rest < 1000) {
				wsb.append(' ').append(ZERO);
			}
			wsb.append(" ").append(UNIT_HIGH[0]);
			sb.append(wsb);
			if (rest > 0) {
				sb.append(' ');
			}
		}
		if (rest > 0) {
			sb.append(section4(rest));
		}
		return sb.toString().trim();
	}

	private static String section(long n) {
		// 0..99999999
		long wan = n / 10000L;
		long rest = n % 10000L;
		StringBuilder sb = new StringBuilder();
		if (wan > 0) sb.append(section4(wan)).append(" wan4");
		if (wan > 0 && rest > 0) sb.append(' ');
		if (rest > 0) sb.append(section4(rest));
		return sb.toString();
	}

	private static String section4(long n) {
		// 0..9999
		StringBuilder sb = new StringBuilder();
		boolean zeroFlag = false;
		int[] digits = new int[4];
		digits[0] = (int) (n / 1000);
		digits[1] = (int) ((n % 1000) / 100);
		digits[2] = (int) ((n % 100) / 10);
		digits[3] = (int) (n % 10);
		// 10/100/1000/10000 的"一"省略：10 → 十（不读 一十）；110 → 一百一十
		for (int i = 0; i < 4; i++) {
			int d = digits[i];
			if (d == 0) {
				zeroFlag = true;
				continue;
			}
			if (zeroFlag && sb.length() > 0) {
				sb.append(' ').append(ZERO).append(' ');
				zeroFlag = false;
			} else {
				zeroFlag = false;
			}
			// 十位特殊：15 → 十五（不加 一）；10 → 十（不加 一）；110 → 一百一十
			if (i == 2 && d == 1) {
				if (sb.length() == 0) {
					sb.append(UNIT_LOW[1]).append(' ');
				} else {
					sb.append(' ').append(UNIT_LOW[1]).append(' ');
				}
			} else if (i == 3 && d == 1 && digits[2] != 0 && digits[2] != 1) {
				// 二十一、三十一：不加 "一"
				sb.append(UNIT_LOW[1]);
			} else {
				sb.append(DIGIT_PINYIN[d]).append(' ');
				if (UNIT_LOW[3 - i].length() > 0) {
					sb.append(UNIT_LOW[3 - i]).append(' ');
				}
			}
		}
		return sb.toString().trim().replaceAll("\\s+", " ");
	}

	// ====================================================================
	// 金额 / 年 / 月 / 日
	// ====================================================================

	private static String moneyToChinese(double value, boolean isUsd) {
		String suffix = isUsd ? DOLLAR : YUAN;
		long yuanPart = (long) value;
		int jiaoPart = (int) Math.round((value - yuanPart) * 10);
		int fenPart = (int) Math.round((value - yuanPart) * 100) % 10;

		StringBuilder sb = new StringBuilder();
		if (yuanPart > 0) {
			sb.append(numToChinese(yuanPart, true)).append(' ').append(suffix);
		}
		if (jiaoPart > 0) {
			if (sb.length() > 0) sb.append(' ');
			sb.append(DIGIT_PINYIN[jiaoPart]).append(' ').append(JIAO);
		}
		if (fenPart > 0) {
			if (sb.length() > 0) sb.append(' ');
			sb.append(DIGIT_PINYIN[fenPart]).append(' ').append(FEN);
		}
		if (sb.length() == 0) {
			sb.append(ZERO).append(' ').append(suffix);
		}
		return sb.toString();
	}

	private static String yearDigitsToChinese(String year4) {
		if (year4.length() != 4) return digitsToChinese(year4);
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < 4; i++) {
			if (i > 0) sb.append(' ');
			int d = year4.charAt(i) - '0';
			sb.append(DIGIT_PINYIN[d]);
		}
		return sb.toString();
	}

	private static String monthToChinese(String m) {
		int v = Integer.parseInt(m);
		return numToChinese(v, false);
	}

	private static String dayToChinese(String d) {
		int v = Integer.parseInt(d);
		// 日的读法：1→ 一；20 → 二十；21 → 二十一；30 → 三十
		if (v < 10) return DIGIT_PINYIN[v];
		return numToChinese(v, false);
	}

	/** 把数字字符串逐位读音（"123" → "yi1 er4 san1"） */
	public static String digitsToChinese(String digits) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < digits.length(); i++) {
			char c = digits.charAt(i);
			if (c < '0' || c > '9') continue;
			if (sb.length() > 0) sb.append(' ');
			sb.append(DIGIT_PINYIN[c - '0']);
		}
		return sb.toString();
	}
}
