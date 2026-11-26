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
package net.dreamlu.mica.ai.detect.core;

import net.dreamlu.mica.ai.detect.model.DetectResult;
import org.opencv.core.Mat;

/**
 * 检测能力抽象：输入一张 BGR 图，返回该图的检测结果。
 *
 * <p>存在的意义是让<b>切片编排逻辑可以脱离 ONNX 单测</b> —— 切片推理的核心
 * （切片网格 → 坐标回映射 → 跨片合并）用假实现即可完整验证，不必加载真模型。
 *
 * <p>{@link DetectEngine} 是生产实现。
 */
public interface DetectSource {

	/**
	 * 检测单张图像。
	 *
	 * @param bgr 原图 BGR Mat；null 或空图时返回 null
	 * @return 检测结果；输入为空时返回 null
	 */
	DetectResult detect(Mat bgr);

}
