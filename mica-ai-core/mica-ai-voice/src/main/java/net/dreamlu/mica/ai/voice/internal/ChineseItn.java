package net.dreamlu.mica.ai.voice.internal;

import lombok.experimental.UtilityClass;

import java.util.*;
import java.util.regex.MatchResult;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 中文数字转阿拉伯数字 (Chinese ITN - Inverse Text Normalization)。
 *
 * <p>对应 Python 端 {@code chinese_itn}：把语音识别出的中文数字转为阿拉伯数字形式。
 * 使用正则表达式进行匹配和替换。
 *
 * <p>用法示例：
 * <pre>{@code
 * String result = ChineseItn.convert("幺九二点幺六八点幺点幺");
 * // → "192.168.1.1"
 * }</pre>
 */
@UtilityClass
public class ChineseItn {
	// ==================== 配置和映射表 ====================

	private static final Map<String, String> UNIT_MAPPING = new LinkedHashMap<>();

	static {
		String[] nullUnits = {"个", "只", "分", "万", "亿", "秒", "年", "月", "日", "天",
			"时", "钟", "人", "层", "楼", "倍", "块", "次"};
		for (String u : nullUnits) UNIT_MAPPING.put(u, null);
		UNIT_MAPPING.put("克", "g");
		UNIT_MAPPING.put("千克", "kg");
		UNIT_MAPPING.put("米", "米");
		UNIT_MAPPING.put("千米", "千米");
		UNIT_MAPPING.put("千米每小时", "km/h");
	}

	private static final List<String> SORTED_UNITS = new ArrayList<>(UNIT_MAPPING.keySet());

	static {
		SORTED_UNITS.sort((a, b) -> b.length() - a.length());
	}

	private static final String COMMON_UNITS;

	static {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < SORTED_UNITS.size(); i++) {
			if (i > 0) sb.append('|');
			sb.append(Pattern.quote(SORTED_UNITS.get(i)));
		}
		COMMON_UNITS = sb.toString();
	}

	private static final Map<Character, Character> NUM_MAPPER = new HashMap<>();

	static {
		NUM_MAPPER.put('零', '0');
		NUM_MAPPER.put('一', '1');
		NUM_MAPPER.put('幺', '1');
		NUM_MAPPER.put('二', '2');
		NUM_MAPPER.put('两', '2');
		NUM_MAPPER.put('三', '3');
		NUM_MAPPER.put('四', '4');
		NUM_MAPPER.put('五', '5');
		NUM_MAPPER.put('六', '6');
		NUM_MAPPER.put('七', '7');
		NUM_MAPPER.put('八', '8');
		NUM_MAPPER.put('九', '9');
		NUM_MAPPER.put('点', '.');
	}

	private static final Map<Character, Integer> VALUE_MAPPER = new HashMap<>();

	static {
		VALUE_MAPPER.put('零', 0);
		VALUE_MAPPER.put('一', 1);
		VALUE_MAPPER.put('二', 2);
		VALUE_MAPPER.put('两', 2);
		VALUE_MAPPER.put('三', 3);
		VALUE_MAPPER.put('四', 4);
		VALUE_MAPPER.put('五', 5);
		VALUE_MAPPER.put('六', 6);
		VALUE_MAPPER.put('七', 7);
		VALUE_MAPPER.put('八', 8);
		VALUE_MAPPER.put('九', 9);
		VALUE_MAPPER.put('十', 10);
		VALUE_MAPPER.put('百', 100);
		VALUE_MAPPER.put('千', 1000);
		VALUE_MAPPER.put('万', 10000);
		VALUE_MAPPER.put('亿', 100000000);
	}

	/** 成语和习语黑名单 */
	private static final Set<String> IDIOMS = new HashSet<>(Arrays.asList(
		"正经八百", "五零二落", "五零四散", "五十步笑百步", "乌七八糟", "污七八糟",
		"四百四病", "思绪万千", "十有八九", "十之八九", "三十而立", "三十六策",
		"三十六计", "三十六行", "三五成群", "三百六十行", "三六九等", "七老八十",
		"七零八落", "七零八碎", "七七八八", "乱七八遭", "乱七八糟", "略知一二",
		"零零星星", "零七八碎", "九九归一", "二三其德", "二三其意", "无银三百两",
		"八九不离十", "百分之百", "年三十", "烂七八糟", "一点一滴", "路易十六",
		"九三学社", "五四运动", "入木三分", "九九八十一", "三七二十一",
		"十二五", "十三五", "十四五", "十五五", "十六五", "十七五", "十八五"
	));

	private static final Pattern FUZZY_REGEX = Pattern.compile("几");

	// ==================== 范围表达式 ====================

	private static final Pattern RANGE_PATTERN_1 = Pattern.compile(
		"([二三四五六七八九])([二三四五六七八九])([十百千万亿])([万千百亿])?");
	private static final Pattern RANGE_PATTERN_2 = Pattern.compile(
		"(十|[一二三四五六七八九十]+[十百千万])([一二三四五六七八九])([一二三四五六七八九])([万千亿])?");
	private static final Pattern RANGE_PATTERN_3 = Pattern.compile(
		"^([一二三四五六七八九])([一二三四五六七八九])$");

	// ==================== 正则表达式模式 ====================

	private static final Pattern UNIT_SUFFIX_PATTERN = Pattern.compile(
		"(" + COMMON_UNITS + "|[a-zA-Z]+)$");

	private static final Pattern PURE_NUM = Pattern.compile(
		"[零幺一二三四五六七八九]+(点[零幺一二三四五六七八九]+)* *([a-zA-Z]|" + COMMON_UNITS + ")?");

	private static final Pattern VALUE_NUM = Pattern.compile(
		"十?(零?[一二两三四五六七八九十][十百千万]{1,2})*零?十?[一二三四五六七八九]?(点[零一二三四五六七八九]+)? *([a-zA-Z]|" + COMMON_UNITS + ")?");

	private static final Pattern PERCENT_VALUE = Pattern.compile(
		"(?<![一二三四五六七八九])(百分之)[零一二三四五六七八九十百千万]+(\\.?[零一二三四五六七八九]+)?");

	private static final Pattern FRACTION_VALUE = Pattern.compile(
		"([零一二三四五六七八九十百千万]+(?:\\.?[零一二三四五六七八九]+)?)\\u5206\\u4e4b([零一二三四五六七八九十百千万]+(?:\\.?[零一二三四五六七八九]+)?)");

	private static final Pattern RATIO_VALUE = Pattern.compile(
		"([零一二三四五六七八九十百千万]+(?:\\.?[零一二三四五六七八九]+)?)\\u6bd4([零一二三四五六七八九十百千万]+(?:\\.?[零一二三四五六七八九]+)?)");

	private static final Pattern TIME_VALUE = Pattern.compile(
		"[零一二两三四五六七八九十]+点([零一二三四五六七八九十]+分)([零一二三四五六七八九十]+秒)?");

	private static final Pattern DATE_VALUE = Pattern.compile(
		"([零一二三四五六七八九十]+年)?([一二三四五六七八九十]+月)?([一二三四五六七八九十]+[日号])?");

	// 主模式（简化版，匹配可能需要转换的中文数字内容）
	private static final Pattern MAIN_PATTERN = Pattern.compile(
		"([a-z]\\s*)?((?:[几零幺一二两三四五六七八九十百千万点比]|[零一二三四五六七八九十] |(?<=[一二两三四五六七八九十])[年月日号分]|(?:分之))+" +
		"(?:[a-zA-Z年月日号]|" + COMMON_UNITS + ")?" +
		"(?:[零幺一二两三四五六七八九十百千万亿点比]|(?:分之))*)");

	/**
	 * 将文本中的中文数字转换为阿拉伯数字。
	 *
	 * @param text 输入文本
	 * @return 转换后的文本
	 */
	public static String convert(String text) {
		if (text == null || text.isEmpty()) return text;
		Matcher m = MAIN_PATTERN.matcher(text);
		StringBuilder sb = new StringBuilder();
		while (m.find()) {
			String replacement = replaceMatch(m, text);
			m.appendReplacement(sb, Matcher.quoteReplacement(replacement));
		}
		m.appendTail(sb);
		return sb.toString();
	}

	private static String replaceMatch(MatchResult match, String fullInput) {
		String fullText = match.group();
		String head = match.group(1);
		String original = match.group(2);
		if (original == null) original = fullText;

		int matchStart = match.start();
		int matchEnd = match.end();

		try {
			// 成语/习语检测（在完整输入字符串中查找）
			int searchStart = Math.max(matchStart - 10, 0);
			int searchEnd = Math.min(matchEnd + 10, fullInput.length());
			String context = fullInput.substring(searchStart, searchEnd);
			for (String idiom : IDIOMS) {
				if (context.contains(idiom)) return fullText;
			}

			// 模糊表达检测
			if (FUZZY_REGEX.matcher(original).find()) return fullText;

			String stripped = stripTrailingUnit(original);

			// 时间
			if (TIME_VALUE.matcher(original).matches()) {
				String result = convertTimeValue(original);
				return (head != null ? head : "") + result;
			}

			// 纯数字
			if (PURE_NUM.matcher(stripped).matches()) {
				String result = convertPureNum(original, false);
				return (head != null ? head : "") + result;
			}

			// 数值
			if (VALUE_NUM.matcher(stripped).matches()) {
				String result = convertValueNum(original);
				return (head != null ? head : "") + result;
			}

			// 百分数
			if (PERCENT_VALUE.matcher(original).matches()) {
				String result = convertPercentValue(original);
				return (head != null ? head : "") + result;
			}

			// 分数
			if (FRACTION_VALUE.matcher(original).matches()) {
				String result = convertFractionValue(original);
				return (head != null ? head : "") + result;
			}

			// 比值
			if (RATIO_VALUE.matcher(original).matches()) {
				String result = convertRatioValue(original);
				return (head != null ? head : "") + result;
			}

			// 日期
			if (DATE_VALUE.matcher(original).matches()) {
				String result = convertDateValue(original);
				return (head != null ? head : "") + result;
			}

		} catch (Exception ignored) {
		}
		return fullText;
	}

	// ==================== 辅助方法 ====================

	private static int chineseDigitToNum(char c) {
		return VALUE_MAPPER.getOrDefault(c, 0);
	}

	static String stripTrailingUnit(String text) {
		Matcher m = UNIT_SUFFIX_PATTERN.matcher(text);
		if (m.find()) {
			return text.substring(0, m.start());
		}
		return text;
	}

	static String[] stripUnit(String original) {
		Pattern unitPattern = Pattern.compile("(" + COMMON_UNITS + ")$");
		Matcher m = unitPattern.matcher(original);
		String stripped;
		String unit;
		if (m.find()) {
			String unitCn = m.group(1);
			stripped = original.substring(0, m.start());
			String mapped = UNIT_MAPPING.get(unitCn);
			unit = (mapped != null) ? mapped : unitCn;
		} else {
			stripped = original;
			unit = "";
		}
		if (unit.isEmpty() && !stripped.isEmpty()) {
			Matcher letterMatch = Pattern.compile("[a-zA-Z]+$").matcher(stripped);
			if (letterMatch.find()) {
				unit = letterMatch.group();
				stripped = stripped.substring(0, letterMatch.start());
			}
		}
		return new String[]{stripped.trim(), unit};
	}

	// ==================== 转换函数 ====================

	static String convertPureNum(String original, boolean strict) {
		String[] parts = stripUnit(original);
		String stripped = parts[0];
		String unit = parts[1];
		if ("一".equals(stripped) && !strict) return original;
		StringBuilder sb = new StringBuilder();
		for (char c : stripped.toCharArray()) {
			sb.append(NUM_MAPPER.getOrDefault(c, c));
		}
		return sb + unit;
	}

	static String convertValueNum(String original) {
		String[] parts = stripUnit(original);
		String stripped = parts[0];
		String unit = parts[1];
		if (!stripped.contains("点")) stripped += "点";
		String[] splitResult = stripped.split("点", 2);
		String intPart = splitResult[0];
		String decimalPart = splitResult.length > 1 ? splitResult[1] : "";
		if (intPart.isEmpty()) return original;

		long value = 0, temp = 0, base = 1;
		for (char c : intPart.toCharArray()) {
			if (c == '十') {
				temp = (temp == 0) ? 10 : VALUE_MAPPER.getOrDefault(c, 10) * temp;
				base = 1;
			} else if (c == '零') {
				base = 1;
			} else if ("一二两三四五六七八九".indexOf(c) >= 0) {
				temp += VALUE_MAPPER.getOrDefault(c, 0);
			} else if (c == '万') {
				value += temp;
				value *= VALUE_MAPPER.getOrDefault(c, 10000);
				base = VALUE_MAPPER.getOrDefault(c, 10000) / 10;
				temp = 0;
			} else if ("百千".indexOf(c) >= 0) {
				value += temp * VALUE_MAPPER.getOrDefault(c, 0);
				base = VALUE_MAPPER.getOrDefault(c, 0) / 10;
				temp = 0;
			}
		}
		value += temp * base;
		String result = String.valueOf(value);

		String decimalStr = convertPureNum(decimalPart, true);
		if (!decimalStr.isEmpty()) {
			result += "." + decimalStr;
		}
		return result + unit;
	}

	static String convertFractionValue(String original) {
		String[] parts = original.split("分之");
		return convertValueNum(parts[0]) + "/" + convertValueNum(parts[1]);
	}

	static String convertPercentValue(String original) {
		return convertValueNum(original.substring(3)) + "%";
	}

	static String convertRatioValue(String original) {
		String[] parts = original.split("比");
		return convertValueNum(parts[0]) + ":" + convertValueNum(parts[1]);
	}

	static String convertTimeValue(String original) {
		String[] parts = original.split("[点分秒]");
		List<String> filtered = new ArrayList<>();
		for (String p : parts) {
			if (!p.isEmpty()) filtered.add(p);
		}
		if (filtered.isEmpty()) return original;
		String hour = convertValueNum(filtered.get(0));
		StringBuilder sb = new StringBuilder();
		sb.append(String.format("%02d", Long.parseLong(hour)));
		if (filtered.size() > 1) {
			String minute = convertValueNum(filtered.get(1));
			sb.append(":").append(String.format("%02d", Long.parseLong(minute)));
		}
		if (filtered.size() > 2) {
			String second = convertValueNum(filtered.get(2));
			sb.append(":").append(String.format("%02d", Long.parseLong(second)));
		}
		return sb.toString();
	}

	static String convertDateValue(String original) {
		StringBuilder sb = new StringBuilder();
		String remaining = original;
		if (remaining.contains("年")) {
			int idx = remaining.indexOf('年');
			String year = remaining.substring(0, idx);
			remaining = remaining.substring(idx + 1);
			sb.append(convertPureNum(year, false)).append("年");
		}
		if (remaining.contains("月")) {
			int idx = remaining.indexOf('月');
			String month = remaining.substring(0, idx);
			remaining = remaining.substring(idx + 1);
			sb.append(convertValueNum(month)).append("月");
		}
		if (remaining.contains("日")) {
			int idx = remaining.indexOf('日');
			String day = remaining.substring(0, idx);
			sb.append(convertValueNum(day)).append("日");
		} else if (remaining.contains("号")) {
			int idx = remaining.indexOf('号');
			String day = remaining.substring(0, idx);
			sb.append(convertValueNum(day)).append("号");
		}
		return sb.toString();
	}
}
