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
package net.dreamlu.mica.ai.textline.model;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

/**
 * 文本行方向：模型输出的 2 个类别。
 *
 * <p>类别顺序来自官方推理包 {@code inference.yml} 的 {@code PostProcess.Topk.label_list}：
 * <pre>
 * label_list:
 * - 0_degree
 * - 180_degree
 * </pre>
 * 即索引 0 为 {@link #DEGREE_0}、索引 1 为 {@link #DEGREE_180}，<b>不要按字典序猜测</b>。
 *
 * <p>{@link #angle} 是「把该文本行转正需要旋转的角度」，可直接交给
 * {@code Imgproc.rotate} 之外的旋转工具使用；{@link #isUpsideDown()} 是其便捷判断。
 */
@Getter
@RequiredArgsConstructor
public enum TextLineOrientation {
	/**
	 * 方向正常（0 度），无需旋转
	 */
	DEGREE_0(0, "0_degree"),
	/**
	 * 倒置（180 度），需要旋转 180 度才可识别
	 */
	DEGREE_180(180, "180_degree");

	/**
	 * 索引 0 对应的方向，即类别 0
	 */
	public static final TextLineOrientation CLASS_0 = DEGREE_0;
	/**
	 * 索引 1 对应的方向，即类别 1
	 */
	public static final TextLineOrientation CLASS_1 = DEGREE_180;

	/**
	 * 把该方向转正需要旋转的角度（0 或 180）
	 */
	private final int angle;
	/**
	 * 官方标签名，与 {@code inference.yml} 的 {@code label_list} 一致
	 */
	private final String label;

	/**
	 * 按模型输出索引解析方向。
	 *
	 * @param index 类别索引（0 或 1）
	 * @return 对应方向
	 * @throws IllegalArgumentException 索引越界
	 */
	public static TextLineOrientation fromClassIndex(int index) {
		switch (index) {
			case 0:
				return DEGREE_0;
			case 1:
				return DEGREE_180;
			default:
				throw new IllegalArgumentException("文本行方向类别索引越界: " + index);
		}
	}

	/**
	 * 是否为倒置方向。
	 *
	 * @return {@code 180 度} 时返回 {@code true}
	 */
	public boolean isUpsideDown() {
		return this == DEGREE_180;
	}
}
