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
 * 活体检测结果。
 *
 * <ul>
 *   <li>{@code liveScore}：真人概率，{@code [0, 1]}，与 {@code LivenessDetector.INDEX_LIVE} 对应</li>
 *   <li>{@code isLive}：{@code liveScore > threshold} 的便捷布尔</li>
 *   <li>{@code attackType}：{@code "real"} / {@code "print"} / {@code "replay"} / {@code "unknown"}</li>
 * </ul>
 *
 * <p>不可变。
 */
@Getter
@AllArgsConstructor
public class LivenessResult {

	private float liveScore;
	private boolean isLive;
	private String attackType;
}