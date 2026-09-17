/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.layout.model;

import lombok.Getter;

/**
 * PP-DocLayoutV2 / V3 文档版面分析的类别枚举（25 类，与官方
 * PaddleOCR/PaddleX {@code paddlex/inference/models/layout_analysis} 字典完全一致）。
 *
 * <p>枚举顺序与官方推理脚本的 {@code label_list} 一致——{@link #getIndex()} 与模型
 * logits 输出槽位顺序一致；用 {@link #of(String)} 做 string → enum 解析。
 *
 * <p>V2 与 V3 共享同一套 25 类字典，差异仅在 V3 多了 Pointer Network 算 reading order。
 */
@Getter
public enum LayoutLabel {

	ABSTRACT("abstract", 0),
	ALGORITHM("algorithm", 1),
	ASIDE_TEXT("aside_text", 2),
	CHART("chart", 3),
	CONTENT("content", 4),
	DISPLAY_FORMULA("display_formula", 5),
	DOC_TITLE("doc_title", 6),
	FIGURE_TITLE("figure_title", 7),
	FOOTER("footer", 8),
	FOOTER_IMAGE("footer_image", 9),
	FOOTNOTE("footnote", 10),
	FORMULA_NUMBER("formula_number", 11),
	HEADER("header", 12),
	HEADER_IMAGE("header_image", 13),
	IMAGE("image", 14),
	INLINE_FORMULA("inline_formula", 15),
	NUMBER("number", 16),
	PARAGRAPH_TITLE("paragraph_title", 17),
	REFERENCE("reference", 18),
	REFERENCE_CONTENT("reference_content", 19),
	SEAL("seal", 20),
	TABLE("table", 21),
	TEXT("text", 22),
	VERTICAL_TEXT("vertical_text", 23),
	VISION_FOOTNOTE("vision_footnote", 24);

	private final String code;
	private final int index;

	LayoutLabel(String code, int index) {
		this.code = code;
		this.index = index;
	}

	public static LayoutLabel of(String code) {
		if (code == null || code.isEmpty()) {
			return null;
		}
		for (LayoutLabel l : values()) {
			if (l.code.equalsIgnoreCase(code)) {
				return l;
			}
		}
		return null;
	}

	public static LayoutLabel ofIndex(int index) {
		if (index < 0 || index >= values().length) {
			return null;
		}
		return values()[index];
	}

	public static int numClasses() {
		return values().length;
	}
}