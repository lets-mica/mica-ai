/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.layout.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Collections;
import java.util.List;

/**
 * 单个版面区域的推理结果。
 *
 * <p>{@code boundingBox} 为原图坐标系下的 {@code [x1, y1, x2, y2]}
 * （已反 letterbox、已过滤、已 clip）；{@code score} 为该类的置信度；
 * {@code readingOrder} 由模型输出第 7 列（order 键）升序解码，0 = 最先读，
 * {@link #READING_ORDER_NONE} 表示模型未给出该列。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LayoutResult {

	private LayoutLabel label;
	private String labelCode;
	private int[] boundingBox;
	private float score;
	private int index;
	private int order;
	private int readingOrder;

	public static List<LayoutResult> emptyList() {
		return Collections.emptyList();
	}

	public static final int READING_ORDER_NONE = -1;
}