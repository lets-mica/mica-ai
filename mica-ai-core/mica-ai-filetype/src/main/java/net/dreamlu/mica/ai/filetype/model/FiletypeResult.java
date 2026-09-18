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
package net.dreamlu.mica.ai.filetype.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import net.dreamlu.mica.ai.filetype.config.PredictionMode;

/**
 * 文件类型识别的最终对外结果。
 *
 * <ul>
 *   <li>{@code outputLabel} —— 经 overwrite_map + 阈值后处理的最终标签</li>
 *   <li>{@code modelLabel} —— 模型直接 argmax 得到的标签（调试用）</li>
 *   <li>{@code score} —— softmax 后该标签的概率，{@code [0, 1]}</li>
 *   <li>{@code contentType} —— 类型元数据（{@link ContentTypeInfo}）</li>
 *   <li>{@code mode} —— 当前预测模式</li>
 *   <li>{@code text} —— 是否文本类型（兜底到 {@code txt} 时为 true）</li>
 * </ul>
 *
 * <p>不可变。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class FiletypeResult {

    private String outputLabel;
    private String modelLabel;
    private float score;
    private ContentTypeInfo contentType;
    private PredictionMode mode;
    private boolean text;

}
