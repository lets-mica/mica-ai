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
package net.dreamlu.mica.ai.captcha.config;

import java.awt.RenderingHints;

/**
 * 验证码图片缩放时使用的插值方式（映射到 {@link RenderingHints}）。
 *
 * <p>ddddocr Python 原版用 PIL 的 {@code ANTIALIAS}（即 LANCZOS）缩放；
 * 本模块默认用 {@link #BILINEAR}，与本项目迁出前的实现一致。
 * 两者在同一张实测图上的识别结果一致（{@code SHTP}），差异只体现在极细笔画上。
 */
public enum CaptchaInterpolation {

	/**
	 * 最近邻插值
	 */
	NEAREST(RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR),

	/**
	 * 双线性插值（默认）
	 */
	BILINEAR(RenderingHints.VALUE_INTERPOLATION_BILINEAR),

	/**
	 * 双三次插值
	 */
	BICUBIC(RenderingHints.VALUE_INTERPOLATION_BICUBIC);

	private final Object awtValue;

	CaptchaInterpolation(Object awtValue) {
		this.awtValue = awtValue;
	}

	/**
	 * 获取对应的 {@link RenderingHints} 常量。
	 *
	 * @return AWT 渲染提示常量
	 */
	public Object getAwtValue() {
		return awtValue;
	}
}
