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
 * 文本行缩放插值方式。
 *
 * <p>纯标记枚举：OpenCV 常量在 {@code net.dreamlu.mica.ai.textline.util} 内映射，
 * 本枚举不持有任何字段，避免核心配置包依赖 OpenCV。
 */
public enum TextLineInterpolation {
	/**
	 * 双线性插值（默认），官方 PaddleX {@code ResizeImage} 的默认行为
	 */
	LINEAR,
	/**
	 * 最近邻，速度最快，适合极小二值文本行
	 */
	NEAREST,
	/**
	 * 三次插值，边缘最锐利
	 */
	CUBIC
}
