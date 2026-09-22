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
package net.dreamlu.mica.ai.textline.config;

/**
 * 输入张量的通道顺序。
 *
 * <p>PaddleX 的 {@code Normalize} 直接对 OpenCV 原生的 <b>BGR</b> 图像按通道归一化
 * （不转 RGB），因此官方参考实现的通道顺序是 {@link #BGR}。
 *
 * <p>⚠️ 实测（2026-09-21，PP-LCNet_x1_0_textline_ori）：本任务判定的是「文字结构朝向」，
 * 是形状线索而非颜色线索，<b>BGR 与 RGB 的判定结果完全一致</b>（干净的合成文本行上概率逐位相同）。
 * 该配置项因此主要面向「换用对通道敏感的模型」的场景，默认按官方约定用 {@link #BGR}。
 */
public enum TextLineChannelOrder {
	/**
	 * BGR 顺序（OpenCV 原生，官方 PaddleX 参考实现，默认）
	 */
	BGR,
	/**
	 * RGB 顺序
	 */
	RGB
}
