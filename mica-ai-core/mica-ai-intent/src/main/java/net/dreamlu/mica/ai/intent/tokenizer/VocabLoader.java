/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent.tokenizer;

import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * BERT vocab.txt 词表加载器。
 *
 * <p>vocab.txt 格式：每行一个 token，行号即为 token ID。
 * <p>与 HuggingFace transformers 的 vocab.txt 完全兼容。
 *
 * <p>内置特殊 token ID：
 * <ul>
 *   <li>[PAD] = 0</li>
 *   <li>[UNK] = 100</li>
 *   <li>[CLS] = 101</li>
 *   <li>[SEP] = 102</li>
 * </ul>
 */
@Slf4j
public final class VocabLoader {

	/** [PAD] token ID */
	public static final int PAD_ID = 0;

	/** [UNK] token ID */
	public static final int UNK_ID = 100;

	/** [CLS] token ID */
	public static final int CLS_ID = 101;

	/** [SEP] token ID */
	public static final int SEP_ID = 102;

	private final Map<String, Integer> tokenToId;
	private final String[] idToToken;
	private final int size;

	private VocabLoader(Map<String, Integer> tokenToId, String[] idToToken) {
		this.tokenToId = tokenToId;
		this.idToToken = idToToken;
		this.size = idToToken.length;
	}

	/**
	 * 从文件路径加载词表。
	 *
	 * @param vocabPath vocab.txt 文件路径
	 * @return VocabLoader 实例
	 * @throws IOException 文件读取失败
	 */
	public static VocabLoader load(Path vocabPath) throws IOException {
		try (BufferedReader reader = Files.newBufferedReader(vocabPath, StandardCharsets.UTF_8)) {
			return loadFromReader(reader);
		}
	}

	/**
	 * 从字符串路径加载词表。
	 *
	 * @param vocabPath vocab.txt 文件路径字符串
	 * @return VocabLoader 实例
	 * @throws IOException 文件读取失败
	 */
	public static VocabLoader load(String vocabPath) throws IOException {
		return load(Path.of(vocabPath));
	}

	/**
	 * 从 classpath 资源加载词表。
	 *
	 * @param classLoader ClassLoader
	 * @param resourcePath 资源路径（如 "vocab.txt"）
	 * @return VocabLoader 实例
	 * @throws IOException 资源读取失败
	 */
	public static VocabLoader loadFromClasspath(ClassLoader classLoader, String resourcePath) throws IOException {
		try (InputStream is = classLoader.getResourceAsStream(resourcePath)) {
			if (is == null) {
				throw new IOException("Resource not found: " + resourcePath);
			}
			try (BufferedReader reader = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
				return loadFromReader(reader);
			}
		}
	}

	private static VocabLoader loadFromReader(BufferedReader reader) throws IOException {
		// 先用列表收集，因为不知道行数
		String line;
		List<String> tokens = new ArrayList<>(21128);
		while ((line = reader.readLine()) != null) {
			// BERT vocab.txt 可能包含多余空白，trim 处理
			line = line.trim();
			if (!line.isEmpty()) {
				tokens.add(line);
			}
		}

		Map<String, Integer> tokenToId = new HashMap<>(tokens.size());
		String[] idToToken = new String[tokens.size()];
		for (int i = 0; i < tokens.size(); i++) {
			String token = tokens.get(i);
			tokenToId.put(token, i);
			idToToken[i] = token;
		}

		log.info("加载词表完成, 共 {} 个 token", tokens.size());
		return new VocabLoader(tokenToId, idToToken);
	}

	/**
	 * 获取 token 对应的 ID。
	 *
	 * @param token token 字符串
	 * @return token ID，未找到时返回 [UNK]=100
	 */
	public int getId(String token) {
		return tokenToId.getOrDefault(token, UNK_ID);
	}

	/**
	 * 获取 ID 对应的 token。
	 *
	 * @param id token ID
	 * @return token 字符串，ID 越界时返回 "[UNK]"
	 */
	public String getToken(int id) {
		if (id >= 0 && id < size) {
			return idToToken[id];
		}
		return "[UNK]";
	}

	/**
	 * 词表大小。
	 */
	public int size() {
		return size;
	}

	/**
	 * 检查 token 是否在词表中。
	 */
	public boolean contains(String token) {
		return tokenToId.containsKey(token);
	}
}
