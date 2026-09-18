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
package net.dreamlu.mica.ai.plate.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 单个车牌识别结果。
 *
 * <ul>
 *   <li>{@code plateCode}：车牌号（双层牌为上下两行拼接）</li>
 *   <li>{@code plateType}：10 类枚举，详见 {@link PlateType}</li>
 *   <li>{@code detectionConfidence}：检测框得分（{@code obj_conf × class score}）</li>
 *   <li>{@code recognitionConfidence}：CTC 解码字符概率均值（双层牌为两行平均）</li>
 *   <li>{@code boundingBox}：原图坐标 {@code [x1, y1, x2, y2]}</li>
 *   <li>{@code landmarks}：4 角点 {@code [4][2]}，顺序 左上 / 右上 / 右下 / 左下</li>
 * </ul>
 *
 * <p>不可变。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class PlateResult {

    private String plateCode;
    private PlateType plateType;
    private float detectionConfidence;
    private float recognitionConfidence;
    private int[] boundingBox;
    private int[][] landmarks;
}