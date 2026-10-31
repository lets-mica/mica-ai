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
package net.dreamlu.mica.ai.common.util;

import lombok.experimental.UtilityClass;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;

/**
 * IO 工具（跨能力通用基础设施）。
 *
 * <p>当前工具方法：
 * <ul>
 *   <li>{@link #readAllBytes(InputStream)} —— 一次性读入流内容</li>
 *   <li>{@link #readAllLines(byte[], Charset)} —— 按行切分文本内容</li>
 * </ul>
 */
@UtilityClass
public class IOUtil {

	public static byte[] readAllBytes(InputStream in) throws IOException {
		ByteArrayOutputStream out = new ByteArrayOutputStream(Math.max(64, in.available()));
		byte[] buf = new byte[8192];
		int n;
		while ((n = in.read(buf)) != -1) {
			out.write(buf, 0, n);
		}
		return out.toByteArray();
	}

	/**
	 * 把文本内容按行切分。
	 *
	 * <p>换行语义与 {@code BufferedReader#readLine()} 一致：{@code \n}、{@code \r}、
	 * {@code \r\n} 均视为行分隔符；末尾换行不产生额外的空行。
	 *
	 * @param content 文本字节内容
	 * @param charset 字符集
	 * @return 按行切分后的列表，内容为空时返回空列表
	 */
	public static List<String> readAllLines(byte[] content, Charset charset) {
		List<String> lines = new ArrayList<>();
		if (content == null || content.length == 0) {
			return lines;
		}
		String text = new String(content, charset);
		int start = 0;
		int len = text.length();
		for (int i = 0; i < len; i++) {
			char c = text.charAt(i);
			if (c == '\n' || c == '\r') {
				lines.add(text.substring(start, i));
				if (c == '\r' && i + 1 < len && text.charAt(i + 1) == '\n') {
					i++;
				}
				start = i + 1;
			}
		}
		if (start < len) {
			lines.add(text.substring(start));
		}
		return lines;
	}
}
