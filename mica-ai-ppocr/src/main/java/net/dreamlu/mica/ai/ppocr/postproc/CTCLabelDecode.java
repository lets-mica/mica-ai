package net.dreamlu.mica.ai.ppocr.postproc;

import lombok.ToString;
import net.dreamlu.mica.ai.ppocr.util.NpUtil;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * CTC greedy decode：argmax → 去连续重复 → 去 blank → 查表出字。
 *
 * <p>对应 Python 端的 {@code CTCLabelDecode}。字符字典从文本文件加载（每行一个字符），
 * 代码会在文件内容前再补一个 {@code "blank"} 占位（与 PaddleX 行为一致）。
 */
@ToString
public final class CTCLabelDecode {

    /** blank 在 vocabulary 中的索引。 */
    public static final int BLANK = 0;

    private final String[] chars;

    public CTCLabelDecode(String characterDictPath) {
        this(Path.of(characterDictPath));
    }

    public CTCLabelDecode(Path characterDictPath) {
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
            // 与 Python `line.rstrip("\n\r")` 行为一致：去除行尾换行（Files.readAllLines 已经做了）
            // 同时去除可能的尾随空白字符
            list.add(stripTrailing(line));
        }
        this.chars = list.toArray(new String[0]);
    }

    /** 词汇表大小（含 blank）。 */
    public int vocabSize() {
        return chars.length;
    }

    /**
     * 对一批索引序列执行 CTC greedy decode。
     *
     * @param indices (batch, seqLen) int 索引
     * @param probs   (batch, seqLen) float 每帧概率（可选，用于置信度）
     * @return {@link Result}：texts 与 scores，长度均为 batch
     */
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

            // 置信度：被保留位置的概率均值
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

    /**
     * 对模型输出执行 CTC decode。
     *
     * @param modelOutput (batch, seqLen, numClasses) 浮点张量（按 C 维在最后）
     * @return {@link Result}
     */
    public Result call(float[][][] modelOutput) {
        int[][] indices = NpUtil.argmaxLastAxis(modelOutput);
        float[][] probs = NpUtil.maxLastAxis(modelOutput);
        return decode(indices, probs);
    }

    /**
     * 解码结果。
     */
    public record Result(String[] texts, float[] scores) {
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
}
