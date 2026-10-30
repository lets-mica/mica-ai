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
package net.dreamlu.mica.ai.classify.config;

/**
 * 输入张量的通道顺序。
 *
 * <p>PaddleClas 的推理流水线为 {@code DecodeImage(to_rgb=True)} → {@code NormalizeImage}，
 * 即先在 <b>RGB</b> 空间归一化，因此 PULC 族的官方约定是 {@link #RGB}。
 *
 * <p>注意这与 {@code mica-ai-textline}（PaddleX 直接对 OpenCV 原生 BGR 归一化）不同，
 * 复用代码时不要照抄那边的默认值。
 */
public enum ClassifyChannelOrder {
	/**
	 * RGB 顺序（PaddleClas 官方约定，默认）
	 */
	RGB,
	/**
	 * BGR 顺序（OpenCV 原生）
	 */
	BGR
}
