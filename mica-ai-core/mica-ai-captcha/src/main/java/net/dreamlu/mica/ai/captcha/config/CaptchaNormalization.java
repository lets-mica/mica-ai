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

/**
 * 验证码模型输入的归一化方式。
 *
 * <p>入参 {@code ddddocr.onnx} 的图头是 {@code DynamicQuantizeLinear}：它按<b>实际张量的
 * min/max</b> 动态把输入量化到 uint8。因此两种归一化在本模型上的 argmax 完全一致
 * （实测同一张验证码图，{@link #DIV_255} 平均置信度 0.99960，
 * {@link #MEAN_0_5} 为 0.99943，CTC 解码结果均为 {@code SHTP}）。
 *
 * <p>差异只体现在 zero-point 偏移上；换用非动态量化的同族模型时，两者的差异会放大，
 * 此时应按该模型训练时的归一化方式选择。
 */
public enum CaptchaNormalization {

	/**
	 * {@code x / 255}，值域 {@code [0, 1]}。
	 *
	 * <p>默认项，与本项目迁出前的实现一致。
	 */
	DIV_255,

	/**
	 * {@code (x - 0.5) / 0.5}，值域 {@code [-1, 1]}。
	 *
	 * <p>ddddocr Python 原版对官方模型采用的归一化。
	 */
	MEAN_0_5;

	/**
	 * 对 {@code [0, 1]} 区间的像素值应用归一化。
	 *
	 * @param value 归一化前的像素值（0~1）
	 * @return 归一化后的值
	 */
	public float apply(float value) {
		switch (this) {
			case MEAN_0_5:
				return (value - 0.5f) / 0.5f;
			case DIV_255:
			default:
				return value;
		}
	}
}
