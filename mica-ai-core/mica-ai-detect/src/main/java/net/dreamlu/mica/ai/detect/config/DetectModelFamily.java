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
package net.dreamlu.mica.ai.detect.config;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * 目标检测模型族。
 *
 * <p>不同模型族的 <b>预处理与后处理契约</b>不同（输入边长、归一化、padding 值、输出张量形态），
 * 因此这里把「契约」显式化为枚举，而不是靠文件名猜。
 *
 * <table border="1">
 *   <caption>族契约对照</caption>
 *   <tr><th>族</th><th>输入</th><th>归一化</th><th>letterbox 填充</th><th>输出</th><th>NMS</th></tr>
 *   <tr>
 *     <td>{@link #NANODET_PLUS}</td><td>416×416</td><td>(x-mean)/std（ImageNet BGR 统计量）</td>
 *     <td>0，INTER_AREA</td><td>6 个张量（stride 8/16/32 × cls/bbox，DFL 解码）</td><td>按类别无关</td>
 *   </tr>
 *   <tr>
 *     <td>{@link #YOLOX_S}</td><td>640×640</td><td>无（原始 0~255）</td>
 *     <td>114，INTER_LINEAR</td><td>1 个 [1,N,85]（cx,cy,w,h,obj,80 类）</td><td>按类别</td>
 *   </tr>
 * </table>
 */
@Getter
@RequiredArgsConstructor
public enum DetectModelFamily {

	/**
	 * NanoDet-Plus（opencv_zoo {@code object_detection_nanodet_2022nov.onnx}，3.6MB）
	 */
	NANODET_PLUS("nanodet_plus", 416),

	/**
	 * YOLOX-s（opencv_zoo {@code object_detection_yolox_2022nov.onnx}，34MB）
	 */
	YOLOX_S("yolox_s", 640);

	/**
	 * 内置 classpath 资源名（{@code mica-ai/models/detect/<resource>.onnx}）
	 */
	private final String resource;

	/**
	 * 模型原生输入边长；配置的 {@code inputSize} 与之不一致时构造期快速失败
	 */
	private final int inputSize;
}
