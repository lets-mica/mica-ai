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
package net.dreamlu.mica.ai.face.verification;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 人脸 1:1 比对结果。
 *
 * <ul>
 *   <li>{@code similarity}：余弦相似度，{@code [-1, 1]}</li>
 *   <li>{@code threshold}：比对阈值（默认 0.35，可按调用方传入覆写）</li>
 *   <li>{@code passed}：{@code similarity >= threshold}</li>
 * </ul>
 *
 * <p>不可变。
 */
@Getter
@AllArgsConstructor
public class VerifyResult {

	private float similarity;
	private float threshold;
	private boolean passed;
}