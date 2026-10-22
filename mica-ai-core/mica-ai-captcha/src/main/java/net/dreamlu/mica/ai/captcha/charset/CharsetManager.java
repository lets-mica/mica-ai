/*
 * Copyright (c) 2019-2029, Dreamlu 卢春梦 (596392912@qq.com & dreamlu.net).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.dreamlu.mica.ai.captcha.charset;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 验证码模型字符集。
 *
 * <p>字符集文件为 UTF-8 文本，<b>每个字符占一个索引</b>，换行符（{@code \n} / {@code \r}）
 * 不占索引。索引 0 固定为 CTC 的 blank（空串），因此模型输出的第 <i>i</i> 类
 * 对应字符集的第 <i>i</i> 个字符。
 *
 * <p><b>刻意不去重</b>：字符集与模型是一一对应的映射表，若文件里出现重复字符而去重，
 * 后续所有索引都会错位，识别结果变成乱码。去重会「看起来更干净」，实则破坏契约。
 *
 * <p>线程安全：构造后不可变。
 */
@Slf4j
public class CharsetManager {

	/**
	 * blank（CTC 空白符）在字符集中的索引
	 */
	public static final int BLANK_INDEX = 0;

	private final List<String> charset;

	private CharsetManager(List<String> charset) {
		this.charset = Collections.unmodifiableList(charset);
	}

	/**
	 * 从外部文件加载字符集。
	 *
	 * @param path 字符集文件路径，支持 {@code classpath:} 前缀
	 * @return 字符集管理器
	 * @throws MicaAiException {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#MODEL_LOAD_FAILED} 读取失败或字符集为空
	 */
	public static CharsetManager load(String path) {
		if (path == null || path.trim().isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "字符集路径不能为空");
		}
		String content = readContent(path);
		List<String> chars = new ArrayList<>(content.length() + 1);
		chars.add("");
		for (int i = 0; i < content.length(); i++) {
			char c = content.charAt(i);
			if (c == '\n' || c == '\r') {
				continue;
			}
			chars.add(String.valueOf(c));
		}
		if (chars.size() <= 1) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"字符集为空: " + path);
		}
		log.info("验证码字符集加载完成: path={} 类别数={}（含 blank）", path, chars.size());
		return new CharsetManager(chars);
	}

	/**
	 * 字符集大小（含 blank），应与模型输出的类别数一致
	 *
	 * @return 类别数
	 */
	public int size() {
		return charset.size();
	}

	/**
	 * 取指定索引对应的字符。
	 *
	 * @param index 类别索引
	 * @return 对应字符；索引越界时返回空串
	 */
	public String charAt(int index) {
		if (index < 0 || index >= charset.size()) {
			return "";
		}
		return charset.get(index);
	}

	/**
	 * 判断索引是否为有效类别。
	 *
	 * @param index 类别索引
	 * @return 在 {@code [0, size())} 区间内返回 {@code true}
	 */
	public boolean isValidIndex(int index) {
		return index >= 0 && index < charset.size();
	}

	private static String readContent(String path) {
		try {
			if (OnnxModelSession.isClasspath(path)) {
				String resource = OnnxModelSession.stripClasspathPrefix(path);
				try (InputStream in = Thread.currentThread().getContextClassLoader()
					.getResourceAsStream(resource)) {
					if (in == null) {
						throw new MicaAiException(ErrorCode.NOT_FOUND,
							"classpath 字符集资源未找到: " + resource);
					}
					return new String(readAll(in), StandardCharsets.UTF_8);
				}
			}
			if (!Files.exists(Paths.get(path))) {
				throw new MicaAiException(ErrorCode.NOT_FOUND, "字符集文件不存在: " + path);
			}
			return new String(Files.readAllBytes(Paths.get(path)), StandardCharsets.UTF_8);
		} catch (IOException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"读取字符集失败: " + path, e);
		}
	}

	private static byte[] readAll(InputStream in) throws IOException {
		java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
		byte[] buffer = new byte[8192];
		int read;
		while ((read = in.read(buffer)) != -1) {
			out.write(buffer, 0, read);
		}
		return out.toByteArray();
	}
}
