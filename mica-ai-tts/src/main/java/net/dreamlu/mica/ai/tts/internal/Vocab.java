package net.dreamlu.mica.ai.tts.internal;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.*;

/**
 * 音素词表管理器和分词器。
 * <p>从 config.json 加载 vocab（音素 → token ID 映射），将音素字符串转换为 token 序列。
 */
public final class Vocab {

	private final Map<Character, Integer> charToId;

	public Vocab(Map<Character, Integer> charToId) {
		this.charToId = charToId;
	}

	/**
	 * 从 config.json 文件加载词表。
	 */
	public static Vocab load(String configPath) throws IOException {
		String content = Files.readString(Path.of(configPath), StandardCharsets.UTF_8);
		Map<Character, Integer> map = parseVocab(content);
		return new Vocab(map);
	}

	/**
	 * 将音素字符串转换为 token ID 列表。
	 */
	public List<Integer> tokenize(String phonemes) {
		List<Integer> tokens = new ArrayList<>();
		for (int i = 0; i < phonemes.length(); i++) {
			char c = phonemes.charAt(i);
			Integer id = charToId.get(c);
			if (id != null) {
				tokens.add(id);
			}
		}
		return tokens;
	}

	/**
	 * 检查字符是否在词表中。
	 */
	public boolean contains(char c) {
		return charToId.containsKey(c);
	}

	/**
	 * 过滤音素字符串，只保留词表中的字符。
	 */
	public String filter(String phonemes) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < phonemes.length(); i++) {
			char c = phonemes.charAt(i);
			if (charToId.containsKey(c)) {
				sb.append(c);
			}
		}
		return sb.toString();
	}

	/**
	 * 简易 JSON vocab 解析器（解析 {"char": int, ...} 格式）。
	 */
	private static Map<Character, Integer> parseVocab(String json) {
		Map<Character, Integer> map = new LinkedHashMap<>();
		// 提取 "vocab" 对象
		int vocabStart = json.indexOf("\"vocab\"");
		if (vocabStart < 0) {
			throw new IllegalArgumentException("Cannot find 'vocab' in config.json");
		}
		int braceStart = json.indexOf('{', vocabStart);
		int braceEnd = findMatchingBrace(json, braceStart);
		String vocabJson = json.substring(braceStart + 1, braceEnd);

		// 解析键值对
		Pattern pattern = Pattern.compile("\"([^\"]+)\"\\s*:\\s*(\\d+)");
		Matcher matcher = pattern.matcher(vocabJson);
		while (matcher.find()) {
			String key = matcher.group(1);
			int value = Integer.parseInt(matcher.group(2));
			// 处理转义字符
			char c = parseJsonChar(key);
			map.put(c, value);
		}
		return map;
	}

	/**
	 * 解析 JSON 字符串中的字符（处理 Unicode 转义）。
	 */
	private static char parseJsonChar(String key) {
		if (key.startsWith("\\u") && key.length() >= 6) {
			int codePoint = Integer.parseInt(key.substring(2, 6), 16);
			return (char) codePoint;
		}
		return key.charAt(0);
	}

	/**
	 * 找到匹配的右大括号位置。
	 */
	private static int findMatchingBrace(String s, int start) {
		int depth = 0;
		for (int i = start; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c == '{') depth++;
			else if (c == '}') {
				depth--;
				if (depth == 0) return i;
			}
		}
		throw new IllegalArgumentException("Unmatched brace at position " + start);
	}
}
