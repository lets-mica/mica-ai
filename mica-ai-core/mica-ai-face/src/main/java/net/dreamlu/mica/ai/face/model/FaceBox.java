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
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 人脸检测框。
 *
 * <p>描述人脸检测模型（YuNet）输出的单个人脸区域，包含边界框坐标、置信度与 5 个关键点
 * (左眼、右眼、鼻尖、左嘴角、右嘴角)。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class FaceBox {

	private float x1;
	private float y1;
	private float x2;
	private float y2;
	private float score;
	private float[][] landmarks;
}