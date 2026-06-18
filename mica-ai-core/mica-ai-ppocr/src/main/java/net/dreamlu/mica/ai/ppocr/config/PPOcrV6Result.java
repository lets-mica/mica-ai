package net.dreamlu.mica.ai.ppocr.config;

import java.util.List;

/**
 * 单条 OCR 识别结果。
 *
 * @param text  识别文本
 * @param score 置信度，范围 [0, 1]
 * @param box   文本框四顶点，顺序：左上、右上、右下、左下
 */
public record PPOcrV6Result(String text, float score, int[][] box) {

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
		return String.format("PPOcrV6Result{text='%s', score=%.4f, box=%s}", text, score, boxAsNestedList());
	}
}
