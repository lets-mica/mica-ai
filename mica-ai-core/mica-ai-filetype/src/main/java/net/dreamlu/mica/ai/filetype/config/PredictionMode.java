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
package net.dreamlu.mica.ai.filetype.config;

/**
 * 预测模式（决定如何处理低于阈值的输出）。
 *
 * <ul>
 *   <li>{@link #HIGH_CONFIDENCE}（默认）—— 按 kb {@code thresholds[label]} 与
 *       {@code medium_confidence_threshold=0.5} 做置信度校验；不通过则按
 *       {@code is_text} 兜底为 {@code txt} / {@code unknown}</li>
 *   <li>{@link #MEDIUM_CONFIDENCE} —— 仅按 {@code medium_confidence_threshold=0.5} 校验</li>
 *   <li>{@link #BEST_GUESS} —— 不做阈值校验，永远采用模型 argmax 输出</li>
 * </ul>
 */
public enum PredictionMode {
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    BEST_GUESS
}
