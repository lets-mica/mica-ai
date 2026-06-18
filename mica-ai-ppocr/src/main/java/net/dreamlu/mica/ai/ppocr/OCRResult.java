package net.dreamlu.mica.ai.ppocr;

import java.util.List;

/**
 * 单条 OCR 识别结果。
 *
 * <p>使用 Java 17 record 表达不可变值对象。{@code box} 为四顶点坐标
 * {@code [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]}，顺序为左上、右上、右下、左下。
 *
 * @param text  识别文本
 * @param score 置信度，范围 [0, 1]
 * @param box   文本框四顶点（int 类型，长度 4×2）
 */
public record OCRResult(String text, float score, int[][] box) {

    /**
     * 返回与 Python 端 {@code box.tolist()} 兼容的 {@code List<List<Integer>>} 表示。
     * 主要用于序列化或日志输出。
     */
    public List<List<Integer>> boxAsNestedList() {
        return List.of(
                List.of(box[0][0], box[0][1]),
                List.of(box[1][0], box[1][1]),
                List.of(box[2][0], box[2][1]),
                List.of(box[3][0], box[3][1])
        );
    }

    @Override
    public String toString() {
        return String.format("OCRResult{text='%s', score=%.4f, box=%s}", text, score, boxAsNestedList());
    }
}
