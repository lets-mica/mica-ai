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
package net.dreamlu.mica.ai.layout.model;

import lombok.Getter;

/**
 * PP-DocLayoutV2 / V3 文档版面分析的类别枚举（25 类，与官方
 * PaddleOCR / PaddleX {@code paddlex/inference/models/layout_analysis} 字典完全一致）。
 *
 * <p>枚举顺序与官方推理脚本的 {@code label_list} 一致——{@code getIndex()} 与模型
 * logits 输出槽位顺序一致；用 {@link #of(String)} 做 string → enum 解析，
 * 用 {@link #ofIndex(int)} 做 index → enum 解析（异常时返回 {@code null}）。
 *
 * <p>V2 与 V3 共享同一套 25 类字典，差异仅在 V3 多了 Pointer Network 算 reading order。
 */
@Getter
public enum LayoutLabel {

	/** 摘要 */
	ABSTRACT("abstract", 0),
	/** 算法 */
	ALGORITHM("algorithm", 1),
	/** 旁注文本 */
	ASIDE_TEXT("aside_text", 2),
	/** 图表 */
	CHART("chart", 3),
	/** 正文 */
	CONTENT("content", 4),
	/** 独立公式（display formula） */
	DISPLAY_FORMULA("display_formula", 5),
	/** 文档标题 */
	DOC_TITLE("doc_title", 6),
	/** 图标题 */
	FIGURE_TITLE("figure_title", 7),
	/** 页脚 */
	FOOTER("footer", 8),
	/** 页脚图片 */
	FOOTER_IMAGE("footer_image", 9),
	/** 脚注 */
	FOOTNOTE("footnote", 10),
	/** 公式编号 */
	FORMULA_NUMBER("formula_number", 11),
	/** 页眉 */
	HEADER("header", 12),
	/** 页眉图片 */
	HEADER_IMAGE("header_image", 13),
	/** 图片 */
	IMAGE("image", 14),
	/** 行内公式 */
	INLINE_FORMULA("inline_formula", 15),
	/** 编号 */
	NUMBER("number", 16),
	/** 段落标题 */
	PARAGRAPH_TITLE("paragraph_title", 17),
	/** 参考文献 */
	REFERENCE("reference", 18),
	/** 参考文献内容 */
	REFERENCE_CONTENT("reference_content", 19),
	/** 印章 */
	SEAL("seal", 20),
	/** 表格 */
	TABLE("table", 21),
	/** 文本段落 */
	TEXT("text", 22),
	/** 竖排文本 */
	VERTICAL_TEXT("vertical_text", 23),
	/** 视觉脚注 */
	VISION_FOOTNOTE("vision_footnote", 24);

	private final String code;
	private final int index;

	LayoutLabel(String code, int index) {
		this.code = code;
		this.index = index;
	}

	/**
	 * 按 code 解析枚举（忽略大小写）。
	 *
	 * @param code 官方字典 code，如 {@code "doc_title"}
	 * @return 对应的枚举；code 为空或未匹配时返回 {@code null}
	 */
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

	/**
	 * 按索引解析枚举。
	 *
	 * @param index 类别索引（与模型 logits 输出槽位一致）
	 * @return 对应的枚举；索引越界时返回 {@code null}
	 */
	public static LayoutLabel ofIndex(int index) {
		if (index < 0 || index >= values().length) {
			return null;
		}
		return values()[index];
	}

	/**
	 * 获取类别总数。
	 *
	 * @return 枚举数量（当前为 25）
	 */
	public static int numClasses() {
		return values().length;
	}
}