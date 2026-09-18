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
package net.dreamlu.mica.ai.face.model;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * 单条人脸特征匹配结果。
 *
 * <ul>
 *   <li>{@code personId}：人脸库侧的业务标识</li>
 *   <li>{@code similarity}：与底库的余弦相似度，{@code [-1, 1]}</li>
 * </ul>
 *
 * <p>不可变。
 */
@Getter
@AllArgsConstructor
public class MatchResult {

	private String personId;
	private float similarity;
}