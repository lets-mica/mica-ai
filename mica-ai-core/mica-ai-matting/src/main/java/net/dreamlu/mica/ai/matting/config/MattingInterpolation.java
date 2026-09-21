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
package net.dreamlu.mica.ai.matting.config;

/**
 * 掩码缩放插值方式。
 *
 * <p>纯标记枚举：OpenCV 常量在 {@code net.dreamlu.mica.ai.matting.util} 内映射，
 * 本枚举不持有任何字段，避免核心配置包依赖 OpenCV。
 */
public enum MattingInterpolation {
	/**
	 * 双线性插值（默认），边缘平滑，适合软 alpha 与放大
	 */
	LINEAR,
	/**
	 * 最近邻，边缘硬、速度最快，适合二值掩码
	 */
	NEAREST,
	/**
	 * 三次插值，边缘最锐利但可能过冲（alpha 会被 clip 回 [0,1]）
	 */
	CUBIC
}
