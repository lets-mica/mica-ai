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
package net.dreamlu.mica.ai.face.quality;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.face.config.FaceQualityConfig;

/**
 * 人脸质量评估结果（eDifFIQA）。
 *
 * <p>{@link #score} 取值范围约 {@code [0, 1]}，越高表示人脸质量越好。
 * LFW（相对高质量）分布普遍在 0.5 以上，XQLFW（混合质量）显著更低。
 *
 * <p>不可变。
 */
@Getter
@RequiredArgsConstructor
public class FaceQualityResult {

	private final float score;

	/**
	 * 人脸质量是否通过预设阈值。
	 *
	 * @param threshold 判定阈值，与 {@link FaceQualityConfig#getThreshold()} 一致
	 * @return {@code score >= threshold} 时 {@code true}
	 */
	public boolean passed(float threshold) {
		return score >= threshold;
	}
}
