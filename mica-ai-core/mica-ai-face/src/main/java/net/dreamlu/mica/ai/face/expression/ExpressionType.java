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
package net.dreamlu.mica.ai.face.expression;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * Progressive Teacher 表情识别的 7 类标签（RAF-DB 数据集）。
 *
 * <p>顺序严格匹配 ONNX 输出的 argmax 索引。
 */
@Getter
@RequiredArgsConstructor
public enum ExpressionType {

	/** 愤怒 */
	ANGRY(0, "愤怒"),
	/** 厌恶 */
	DISGUST(1, "厌恶"),
	/** 恐惧 */
	FEARFUL(2, "恐惧"),
	/** 开心 */
	HAPPY(3, "开心"),
	/** 平静 */
	NEUTRAL(4, "平静"),
	/** 悲伤 */
	SAD(5, "悲伤"),
	/** 惊讶 */
	SURPRISED(6, "惊讶");

	private final int index;
	private final String desc;

	public static ExpressionType fromIndex(int index) {
		if (index < 0 || index >= 7) {
			throw new IllegalArgumentException("表情索引越界: " + index);
		}
		ExpressionType[] values = values();
		return values[index];
	}
}
