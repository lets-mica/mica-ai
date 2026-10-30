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
 * 证件照尺寸规格。
 *
 * <p>毫米数按二代身份证 / 护照通用规格，{@link #widthPx(int)} / {@link #heightPx(int)}
 * 按给定 DPI 换算成像素（300 DPI 打印对应：
 * 小一寸 260×378、一寸 295×413、大一寸 390×567、
 * 小二寸 413×531、二寸 413×579、大两寸 413×626）。
 *
 * @author DreamLu
 */
@Getter
public enum IdPhotoSpec {

	/** 小一寸 22×32mm */
	SMALL_ONE("小一寸", 22, 32),

	/** 一寸 25×35mm */
	ONE("一寸", 25, 35),

	/** 大一寸 33×48mm */
	LARGE_ONE("大一寸", 33, 48),

	/** 小二寸 35×45mm */
	SMALL_TWO("小二寸", 35, 45),

	/** 二寸 35×49mm */
	TWO("二寸", 35, 49),

	/** 大两寸 35×53mm */
	LARGE_TWO("大两寸", 35, 53);

	/** 规格中文名 */
	private final String label;

	/** 宽（毫米） */
	private final int widthMm;

	/** 高（毫米） */
	private final int heightMm;

	IdPhotoSpec(String label, int widthMm, int heightMm) {
		this.label = label;
		this.widthMm = widthMm;
		this.heightMm = heightMm;
	}

	/**
	 * 毫米转像素。
	 *
	 * @param dpi 目标分辨率
	 * @return 像素宽度
	 */
	public int widthPx(int dpi) {
		return Math.round(widthMm / 25.4f * dpi);
	}

	/**
	 * 毫米转像素。
	 *
	 * @param dpi 目标分辨率
	 * @return 像素高度
	 */
	public int heightPx(int dpi) {
		return Math.round(heightMm / 25.4f * dpi);
	}

	/**
	 * 宽高比。
	 *
	 * @return 宽 / 高
	 */
	public float aspect() {
		return (float) widthMm / (float) heightMm;
	}

	/**
	 * 按名称或枚举名解析规格。
	 *
	 * @param name 规格名，如 {@code ONE} 或 {@code 一寸}
	 * @return 对应规格；无法解析时返回 null
	 */
	public static IdPhotoSpec of(String name) {
		if (name == null || name.trim().isEmpty()) {
			return null;
		}
		String key = name.trim();
		for (IdPhotoSpec spec : values()) {
			if (spec.name().equalsIgnoreCase(key) || spec.label.equals(key)) {
				return spec;
			}
		}
		return null;
	}

}
