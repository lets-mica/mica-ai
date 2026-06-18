package net.dreamlu.mica.ai.tts.engine;

import net.dreamlu.mica.ai.tts.config.KokoroTtsConfig;

import java.util.*;
import java.util.regex.*;

/**
 * 文本前端处理器。
 * <p>负责文本预处理：分句、音素长度限制、批处理。
 */
public final class TextFrontend {
	private static final Pattern SPLIT_PATTERN = Pattern.compile("([.,!?;，。！？；])");

	/**
	 * 将音素字符串按标点分割成多个批次，每批不超过 MAX_PHONEME_LENGTH。
	 *
	 * @param phonemes 音素字符串
	 * @return 分批后的音素列表
	 */
	public static List<String> splitPhonemes(String phonemes) {
		String[] parts = SPLIT_PATTERN.split(phonemes);
		List<String> batches = new ArrayList<>();
		StringBuilder current = new StringBuilder();

		Matcher matcher = SPLIT_PATTERN.matcher(phonemes);
		List<String> segments = new ArrayList<>();
		int lastEnd = 0;

		while (matcher.find()) {
			String segment = phonemes.substring(lastEnd, matcher.end());
			segments.add(segment);
			lastEnd = matcher.end();
		}
		if (lastEnd < phonemes.length()) {
			segments.add(phonemes.substring(lastEnd));
		}

		for (String segment : segments) {
			String trimmed = segment.trim();
			if (trimmed.isEmpty()) continue;

			if (current.length() + trimmed.length() + 1 >= KokoroTtsConfig.MAX_PHONEME_LENGTH) {
				if (!current.isEmpty()) {
					batches.add(current.toString().trim());
				}
				current = new StringBuilder(trimmed);
			} else {
				if (!current.isEmpty()) {
					current.append(" ");
				}
				current.append(trimmed);
			}
		}

		if (!current.isEmpty()) {
			batches.add(current.toString().trim());
		}

		return batches;
	}

	/**
	 * 简单的音频静音裁剪（去除首尾静音）。
	 *
	 * @param audio 音频数据
	 * @param threshold 静音阈值（默认 0.01f）
	 * @return 裁剪后的音频
	 */
	public static float[] trimSilence(float[] audio, float threshold) {
		if (audio == null || audio.length == 0) return audio;

		int start = 0;
		int end = audio.length - 1;

		// 找到第一个非静音样本
		while (start < audio.length && Math.abs(audio[start]) < threshold) {
			start++;
		}

		// 找到最后一个非静音样本
		while (end >= 0 && Math.abs(audio[end]) < threshold) {
			end--;
		}

		if (start > end) {
			return new float[0];
		}

		// 前后各保留一小段缓冲
		start = Math.max(0, start - 100);
		end = Math.min(audio.length - 1, end + 100);

		int len = end - start + 1;
		float[] trimmed = new float[len];
		System.arraycopy(audio, start, trimmed, 0, len);
		return trimmed;
	}
}
