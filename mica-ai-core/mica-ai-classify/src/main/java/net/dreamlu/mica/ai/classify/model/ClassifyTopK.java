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
package net.dreamlu.mica.ai.classify.model;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * Top-K 分类项：类别索引 + 标签 + 概率。
 *
 * <p>{@link #index} 是模型输出原始索引，与 label_list 行序一致；
 * {@link #label} 是该索引对应的标签；{@link #score} 为 softmax / sigmoid 概率。
 */
@Getter
@ToString
@RequiredArgsConstructor
public class ClassifyTopK {

	/**
	 * 类别索引（模型输出顺序）
	 */
	private final int index;
	/**
	 * 标签（来自 label_list）
	 */
	private final String label;
	/**
	 * 概率（softmax 归一化后，或 sigmoid 独立概率）
	 */
	private final float score;

}
