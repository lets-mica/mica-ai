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

	/**
	 * 返回共享的不可变空结果列表。
	 *
	 * @return 空的 {@code List<LayoutResult>}
	 */
	public static List<LayoutResult> emptyList() {
		return Collections.emptyList();
	}

	/**
	 * 阅读顺序缺失时 readingOrder 字段的占位值。
	 */
	public static final int READING_ORDER_NONE = -1;
}
