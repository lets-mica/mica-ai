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
 * 表情识别结果（7 类 softmax 概率）。
 *
 * <p>不可变。
 */
@Getter
@RequiredArgsConstructor
public class ExpressionResult {

	private final ExpressionType expression;
	private final float score;
	private final float[] probabilities;
}
