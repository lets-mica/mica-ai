package net.dreamlu.mica.ai.ppocr.postprocessor;

import lombok.ToString;
import net.dreamlu.mica.ai.ppocr.utils.NdArrayUtils;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * CTC greedy decode：argmax → 去连续重复 → 去 blank → 查表出字。
 */
@ToString
public final class CtcLabelDecoder {

	public static final int BLANK = 0;

	private final String[] chars;

	public CtcLabelDecoder(String characterDictPath) {
		this(Path.of(characterDictPath));
	}

	public CtcLabelDecoder(Path characterDictPath) {
		if (!Files.isReadable(characterDictPath)) {
			throw new IllegalArgumentException(
				"字符字典不可读: " + characterDictPath.toAbsolutePath());
		}
		List<String> lines;
		try {
			lines = Files.readAllLines(characterDictPath, StandardCharsets.UTF_8);
		} catch (IOException e) {
			throw new RuntimeException("读取字符字典失败: " + characterDictPath, e);
		}

		List<String> list = new ArrayList<>(lines.size() + 1);
		list.add("blank");
		for (String line : lines) {
			list.add(stripTrailing(line));
		}
		this.chars = list.toArray(new String[0]);
	}

	private static String stripTrailing(String s) {
		if (s == null) return "";
		int end = s.length();
		while (end > 0) {
			char c = s.charAt(end - 1);
			if (c == '\n' || c == '\r' || c == ' ' || c == '\t') {
				end--;
			} else {
				break;
			}
		}
		return s.substring(0, end);
	}

	public int vocabSize() {
		return chars.length;
	}

	public Result decode(int[][] indices, float[][] probs) {
		int b = indices.length;
		String[] texts = new String[b];
		float[] scores = new float[b];
		for (int i = 0; i < b; i++) {
			int[] seq = indices[i];
			int t = seq.length;
			boolean[] keep = new boolean[t];
			if (t > 0) {
				keep[0] = true;
				for (int j = 1; j < t; j++) {
					keep[j] = seq[j] != seq[j - 1];
				}
				for (int j = 0; j < t; j++) {
					if (seq[j] == BLANK) {
						keep[j] = false;
					}
				}
			}

			StringBuilder sb = new StringBuilder();
			for (int j = 0; j < t; j++) {
				if (keep[j]) {
					int idx = seq[j];
					if (idx >= 0 && idx < chars.length) {
						sb.append(chars[idx]);
					}
				}
			}
			texts[i] = sb.toString();

			if (probs == null) {
				scores[i] = 1.0f;
			} else {
				float sum = 0f;
				int count = 0;
				for (int j = 0; j < t; j++) {
					if (keep[j]) {
						sum += probs[i][j];
						count++;
					}
				}
				scores[i] = count > 0 ? sum / count : 0.0f;
			}
		}
		return new Result(texts, scores);
	}

	public Result call(float[][][] modelOutput) {
		int[][] indices = NdArrayUtils.argmaxLastAxis(modelOutput);
		float[][] probs = NdArrayUtils.maxLastAxis(modelOutput);
		return decode(indices, probs);
	}

	public record Result(String[] texts, float[] scores) {}
}
