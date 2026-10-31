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
package net.dreamlu.mica.ai.captcha.model;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.ToString;

/**
 * 验证码识别结果。
 *
 * <p>{@link #text} 是 CTC 解码后的最终文本；{@link #confidence} 是<b>已输出字符</b>
 * （不含 blank）的平均 softmax 置信度；{@link #charConfidences} 给出逐字符置信度，
 * 可用于「置信度太低就认为识别失败」的二次校验。
 *
 * <p>本结果不可变，不持有任何需要释放的原生资源。
 */
@Getter
@ToString
@RequiredArgsConstructor
public class CaptchaResult {

	/**
	 * 识别文本。
	 */
	private final String text;
	/**
	 * 平均置信度。
	 */
	private final float confidence;
	/**
	 * 逐字符置信度。
	 */
	private final float[] charConfidences;

	/**
	 * 识别结果是否为空（未输出任何字符）。
	 *
	 * @return 文本为空时返回 {@code true}
	 */
	public boolean isEmpty() {
		return text == null || text.isEmpty();
	}

	/**
	 * 判断置信度是否达到给定阈值。
	 *
	 * @param threshold 阈值
	 * @return 置信度不低于阈值时返回 {@code true}
	 */
	public boolean isConfident(float threshold) {
		return confidence >= threshold;
	}

}
