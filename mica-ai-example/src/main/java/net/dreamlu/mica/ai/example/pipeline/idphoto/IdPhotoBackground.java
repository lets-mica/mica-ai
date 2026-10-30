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
package net.dreamlu.mica.ai.example.pipeline.idphoto;

import lombok.Getter;

/**
 * 证件照底色。
 *
 * <p>底色按 <b>RGB</b> 给出（{@code mica-ai-matting} / {@code mica-ai-humanseg} 的
 * {@code compositeOnColor} 同样是 RGB 入参，内部再转成 OpenCV 的 BGR）。
 *
 * @author DreamLu
 */
@Getter
public enum IdPhotoBackground {

	/** 白底 255,255,255 */
	WHITE("白底", 255, 255, 255),

	/** 蓝底 67,142,219 */
	BLUE("蓝底", 67, 142, 219),

	/** 红底 255,0,0 */
	RED("红底", 255, 0, 0);

	/** 底色中文名 */
	private final String label;

	/** 红 */
	private final int r;

	/** 绿 */
	private final int g;

	/** 蓝 */
	private final int b;

	IdPhotoBackground(String label, int r, int g, int b) {
		this.label = label;
		this.r = r;
		this.g = g;
		this.b = b;
	}

	/**
	 * RGB 三元组。
	 *
	 * @return {@code [r, g, b]}
	 */
	public int[] rgb() {
		return new int[]{r, g, b};
	}

	/**
	 * 按名称或枚举名解析底色。
	 *
	 * @param name 底色名，如 {@code BLUE} 或 {@code 蓝底}
	 * @return 对应底色；无法解析时返回 null
	 */
	public static IdPhotoBackground of(String name) {
		if (name == null || name.trim().isEmpty()) {
			return null;
		}
		String key = name.trim();
		for (IdPhotoBackground background : values()) {
			if (background.name().equalsIgnoreCase(key) || background.label.equals(key)) {
				return background;
			}
		}
		return null;
	}

}
