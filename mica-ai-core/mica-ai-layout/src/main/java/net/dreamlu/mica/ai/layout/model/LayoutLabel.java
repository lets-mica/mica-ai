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

import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Set;

/**
 * PP-DocLayoutV2 / V3 文档版面分析的类别枚举（25 类，与官方
 * PaddleOCR / PaddleX {@code paddlex/inference/models/layout_analysis} 字典完全一致）。
 *
 * <p>枚举顺序与官方推理脚本的 {@code label_list} 一致——{@code getIndex()} 与模型
 * logits 输出槽位顺序一致；用 {@link #of(String)} 做 string → enum 解析，
 * 用 {@link #ofIndex(int)} 做 index → enum 解析（异常时返回 {@code null}）。
 *
 * <p>V2 与 V3 共享同一套 25 类字典，差异仅在 V3 多了 Pointer Network 算 reading order。
 *
 * <p>{@link #isSkipOrder()} 对齐 PaddleX {@code LayoutAnalysisProcess} 的
 * {@code SKIP_ORDER_LABELS} 默认名单（11 类）：这些标签**不参与阅读顺序编号**。
 * 需要与 PaddleX 的 {@code order} 字段逐值一致时按此名单过滤，
 * 或通过 {@code LayoutConfig.skipOrderLabels} 整体覆盖。
 */
@Getter
public enum LayoutLabel {

	/** 摘要 */
	ABSTRACT("abstract", 0, false),
	/** 算法 */
	ALGORITHM("algorithm", 1, false),
	/** 旁注文本 */
	ASIDE_TEXT("aside_text", 2, true),
	/** 图表 */
	CHART("chart", 3, true),
	/** 正文 */
	CONTENT("content", 4, false),
	/** 独立公式（display formula） */
	DISPLAY_FORMULA("display_formula", 5, false),
	/** 文档标题 */
	DOC_TITLE("doc_title", 6, false),
	/** 图标题 */
	FIGURE_TITLE("figure_title", 7, true),
	/** 页脚 */
	FOOTER("footer", 8, true),
	/** 页脚图片 */
	FOOTER_IMAGE("footer_image", 9, true),
	/** 脚注 */
	FOOTNOTE("footnote", 10, true),
	/** 公式编号 */
	FORMULA_NUMBER("formula_number", 11, false),
	/** 页眉 */
	HEADER("header", 12, true),
	/** 页眉图片 */
	HEADER_IMAGE("header_image", 13, true),
	/** 图片 */
	IMAGE("image", 14, true),
	/** 行内公式 */
	INLINE_FORMULA("inline_formula", 15, false),
	/** 编号 */
	NUMBER("number", 16, false),
	/** 段落标题 */
	PARAGRAPH_TITLE("paragraph_title", 17, false),
	/** 参考文献 */
	REFERENCE("reference", 18, false),
	/** 参考文献内容 */
	REFERENCE_CONTENT("reference_content", 19, false),
	/** 印章 */
	SEAL("seal", 20, false),
	/** 表格 */
	TABLE("table", 21, true),
	/** 文本段落 */
	TEXT("text", 22, false),
	/** 竖排文本 */
	VERTICAL_TEXT("vertical_text", 23, false),
	/** 视觉脚注 */
	VISION_FOOTNOTE("vision_footnote", 24, true);

	private final String code;
	private final int index;
	private final boolean skipOrder;

	LayoutLabel(String code, int index, boolean skipOrder) {
		this.code = code;
		this.index = index;
		this.skipOrder = skipOrder;
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

	/**
	 * 获取 PaddleX 默认的「不参与阅读顺序编号」标签名单。
	 *
	 * <p>对应 PaddleX {@code LayoutAnalysisProcess.SKIP_ORDER_LABELS} 的 11 类：
	 * {@code figure_title / vision_footnote / image / chart / table / header /
	 * header_image / footer / footer_image / footnote / aside_text}。
	 *
	 * @return 不可变的标签名单（按枚举声明顺序）
	 */
	public static Set<LayoutLabel> defaultSkipOrderLabels() {
		Set<LayoutLabel> set = new LinkedHashSet<>();
		for (LayoutLabel l : values()) {
			if (l.skipOrder) {
				set.add(l);
			}
		}
		return Collections.unmodifiableSet(set);
	}
}