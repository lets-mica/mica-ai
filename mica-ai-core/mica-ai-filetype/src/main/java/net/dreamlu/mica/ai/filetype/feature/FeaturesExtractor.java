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
package net.dreamlu.mica.ai.filetype.feature;

import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.filetype.config.ModelConfig;

import java.util.Arrays;

/**
 * Magika features v2 特征提取（对齐 Python {@code _extract_features_from_seekable}）。
 *
 * <p>输入为已按 {@code block_size} 截断的头部与尾部原始字节：
 * 头部 lstrip 后取前 {@code beg_size} 字节左对齐写入，尾部 rstrip 后取末
 * {@code end_size} 字节右对齐写入，不足部分填充 {@code padding_token}。
 * 文件大小 &le; block_size 时调用方应把头尾传入同一份原始内容。
 *
 * <p>输出张量形状 {@code int[beg_size + end_size]}，可直接喂给 ONNX 模型。
 */
@UtilityClass
public class FeaturesExtractor {

	private static final boolean[] ASCII_WHITESPACE = new boolean[256];

	static {
		ASCII_WHITESPACE[0x09] = true;
		ASCII_WHITESPACE[0x0A] = true;
		ASCII_WHITESPACE[0x0B] = true;
		ASCII_WHITESPACE[0x0C] = true;
		ASCII_WHITESPACE[0x0D] = true;
		ASCII_WHITESPACE[0x20] = true;
	}

	public static int[] extract(ModelConfig config, byte[] head, byte[] tail) {
		int begSize = config.getBegSize();
		int endSize = config.getEndSize();
		int padding = config.getPaddingToken();
		int total = config.featuresSize();
		int[] features = new int[total];
		Arrays.fill(features, padding);

		int begStart = lstrip(head);
		int begAvailable = head.length - begStart;
		int begLen = Math.min(begSize, begAvailable);
		for (int i = 0; i < begLen; i++) {
			features[i] = head[begStart + i] & 0xFF;
		}

		int endLimit = rstrip(tail);
		int endLen = Math.min(endSize, endLimit);
		for (int i = 0; i < endLen; i++) {
			features[total - endLen + i] = tail[endLimit - endLen + i] & 0xFF;
		}
		return features;
	}

	public static boolean hasEnoughMeaningfulBytes(ModelConfig config, int[] features) {
		int idx = config.getMinFileSizeForDl() - 1;
		if (idx < 0) {
			return true;
		}
		return features[idx] != config.getPaddingToken();
	}

	private static int lstrip(byte[] content) {
		int i = 0;
		while (i < content.length && isAsciiWhitespace(content[i])) {
			i++;
		}
		return i;
	}

	private static int rstrip(byte[] content) {
		int i = content.length;
		while (i > 0 && isAsciiWhitespace(content[i - 1])) {
			i--;
		}
		return i;
	}

	private static boolean isAsciiWhitespace(byte b) {
		int v = b & 0xFF;
		return v <= 32 && ASCII_WHITESPACE[v];
	}
}
