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

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Collections;
import java.util.List;

/**
 * 单个版面区域的推理结果。
 *
 * <p>{@code boundingBox} 为原图坐标系下的 {@code [x1, y1, x2, y2]}
 * （已反 letterbox、已过滤、已 clip）；{@code score} 为该类的置信度。
 *
 * <p>{@code readingOrder} 对齐 PaddleX {@code order} 字段语义：
 * <b>从 1 开始</b>顺序编号，且 {@link LayoutLabel#isSkipOrder() 跳过类}
 * （页眉页脚 / 图片 / 表格 / 图标题等 11 类）**不占用编号**、固定为
 * {@link #NO_READING_ORDER}。
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

	/**
	 * 返回共享的不可变空结果列表。
	 *
	 * @return 空的 {@code List<LayoutResult>}
	 */
	public static List<LayoutResult> emptyList() {
		return Collections.emptyList();
	}

	/**
	 * 「无阅读顺序」占位值（对齐 PaddleX {@code order = None}）。
	 *
	 * <p>两种情况取该值，调用方**无需区分**（都是「该项不属于线性阅读序列」）：
	 * <ul>
	 *   <li>标签在 {@code skipOrderLabels} 名单内（页眉页脚 / 图片 / 表格 / 图标题等）</li>
	 *   <li>模型未输出 order 列（V2 模型，或导出形态变化）</li>
	 * </ul>
	 *
	 * <p>区分方式：{@code LayoutResult.getLabel().isSkipOrder()} 为真即属第一种；
	 * 整张图所有项都为该值则说明是第二种。
	 */
	public static final int NO_READING_ORDER = -1;
}
