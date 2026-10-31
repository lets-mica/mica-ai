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
import lombok.ToString;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 图像分类结果。
 *
 * <p>{@link #getTopK()} 按概率降序给出前 K 项；{@link #getScores()} 保留<b>原始索引顺序</b>
 * 的全部概率，便于多标签场景（如 {@code table_attribute}）按属性名直接取值。
 *
 * <p>本结果不持有任何 OpenCV {@code Mat}（不持有原生内存），因此<b>无需 close</b>，
 * 可自由缓存与传递。
 */
@Getter
@ToString
public class ClassifyResult {

	private final List<ClassifyTopK> topK;
	private final float[] scores;
	private final String[] labels;
	private final boolean multiLabel;

	public ClassifyResult(List<ClassifyTopK> topK, float[] scores,
						  String[] labels, boolean multiLabel) {
		this.topK = topK == null ? Collections.emptyList() : topK;
		this.scores = scores;
		this.labels = labels;
		this.multiLabel = multiLabel;
	}

	/**
	 * 概率最高的分类项。
	 *
	 * @return 最高项；结果为空时返回 {@code null}
	 */
	public ClassifyTopK top() {
		return topK.isEmpty() ? null : topK.get(0);
	}

	/**
	 * 最高概率对应的标签。
	 *
	 * @return 标签；无结果时返回 {@code null}
	 */
	public String topLabel() {
		ClassifyTopK item = top();
		return item == null ? null : item.getLabel();
	}

	/**
	 * 最高概率。
	 *
	 * @return 概率；无结果时返回 {@code 0}
	 */
	public float topScore() {
		ClassifyTopK item = top();
		return item == null ? 0f : item.getScore();
	}

	/**
	 * 类别总数。
	 *
	 * @return 类别数
	 */
	public int classCount() {
		return scores == null ? 0 : scores.length;
	}

	/**
	 * 按索引取概率。
	 *
	 * @param index 类别索引
	 * @return 概率；越界时返回 {@code 0}
	 */
	public float scoreAt(int index) {
		if (scores == null || index < 0 || index >= scores.length) {
			return 0f;
		}
		return scores[index];
	}

	/**
	 * 按标签取概率（标签重复时返回第一个匹配项，与 label_list 的索引语义一致）。
	 *
	 * @param label 标签
	 * @return 概率；无匹配时返回 {@code 0}
	 */
	public float scoreOf(String label) {
		if (labels == null || scores == null || label == null) {
			return 0f;
		}
		for (int i = 0; i < labels.length && i < scores.length; i++) {
			if (label.equals(labels[i])) {
				return scores[i];
			}
		}
		return 0f;
	}

	/**
	 * 多标签结果对所有类别做一次阈值过滤。
	 *
	 * <p>仅多标签场景有意义（各属性独立）；返回顺序与 label_list 一致。
	 *
	 * @param threshold 判定阈值
	 * @return 命中的分类项
	 */
	public List<ClassifyTopK> labelsAbove(float threshold) {
		List<ClassifyTopK> hit = new ArrayList<>();
		if (scores == null || labels == null) {
			return hit;
		}
		int n = Math.min(scores.length, labels.length);
		for (int i = 0; i < n; i++) {
			if (scores[i] >= threshold) {
				hit.add(new ClassifyTopK(i, labels[i], scores[i]));
			}
		}
		return hit;
	}

}
