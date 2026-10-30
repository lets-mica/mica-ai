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
package net.dreamlu.mica.ai.classify.postprocess;

import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.classify.model.ClassifyTopK;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 分类后处理：softmax / sigmoid / Top-K，纯 Java，无状态。
 */
@UtilityClass
public class ClassifyPostProcessor {

	/**
	 * 对 logits 做数值稳定的 softmax。
	 *
	 * @param logits 原始 logits
	 * @return 概率数组（与入参等长，和为 1）
	 */
	public static float[] softmax(float[] logits) {
		if (logits == null || logits.length == 0) {
			return new float[0];
		}
		float max = logits[0];
		for (float v : logits) {
			if (v > max) {
				max = v;
			}
		}
		double sum = 0d;
		double[] exp = new double[logits.length];
		for (int i = 0; i < logits.length; i++) {
			exp[i] = Math.exp(logits[i] - max);
			sum += exp[i];
		}
		float[] out = new float[logits.length];
		for (int i = 0; i < logits.length; i++) {
			out[i] = (float) (exp[i] / sum);
		}
		return out;
	}

	/**
	 * 对 logits 逐元素做 sigmoid（多标签场景）。
	 *
	 * @param logits 原始 logits
	 * @return 各标签独立概率
	 */
	public static float[] sigmoid(float[] logits) {
		if (logits == null || logits.length == 0) {
			return new float[0];
		}
		float[] out = new float[logits.length];
		for (int i = 0; i < logits.length; i++) {
			out[i] = (float) (1d / (1d + Math.exp(-logits[i])));
		}
		return out;
	}

	/**
	 * 按概率降序取前 K 项；同分时按索引升序，保证结果稳定。
	 *
	 * @param scores 各索引概率（保持原始索引顺序）
	 * @param labels 与索引对应的标签
	 * @param topK   期望返回的项数
	 * @return Top-K 列表，长度不超过 {@code min(topK, scores.length)}
	 */
	public static List<ClassifyTopK> topK(float[] scores, String[] labels, int topK) {
		if (scores == null || scores.length == 0) {
			return Collections.emptyList();
		}
		int size = Math.min(Math.max(topK, 1), scores.length);
		List<Integer> indices = new ArrayList<>(scores.length);
		for (int i = 0; i < scores.length; i++) {
			indices.add(i);
		}
		indices.sort((a, b) -> {
			int cmp = Float.compare(scores[b], scores[a]);
			return cmp != 0 ? cmp : Integer.compare(a, b);
		});
		List<ClassifyTopK> items = new ArrayList<>(size);
		for (int i = 0; i < size; i++) {
			int index = indices.get(i);
			String label = labels != null && index < labels.length ? labels[index] : String.valueOf(index);
			items.add(new ClassifyTopK(index, label, scores[index]));
		}
		return items;
	}
}
