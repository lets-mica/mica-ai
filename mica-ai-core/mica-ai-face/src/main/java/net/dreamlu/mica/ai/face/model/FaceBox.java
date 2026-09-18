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
 * 人脸检测框。
 *
 * <p>描述人脸检测模型（YuNet）输出的单个人脸区域，字段含义：
 * <ul>
 *   <li>{@code x1 / y1 / x2 / y2}：原图坐标系下的边界框（像素）</li>
 *   <li>{@code score}：YuNet 的 sqrt(cls × obj) 综合分，{@code [0, 1]}</li>
 *   <li>{@code landmarks}：5 个关键点 {@code float[5][2]}，顺序为
 *       左眼 / 右眼 / 鼻尖 / 左嘴角 / 右嘴角；可为 {@code null}（仅 YuNet 部分 stride 输出时）</li>
 * </ul>
 *
 * <p>不可变。
 */
@Getter
@AllArgsConstructor
public class FaceBox {

	private float x1;
	private float y1;
	private float x2;
	private float y2;
	private float score;
	private float[][] landmarks;
}