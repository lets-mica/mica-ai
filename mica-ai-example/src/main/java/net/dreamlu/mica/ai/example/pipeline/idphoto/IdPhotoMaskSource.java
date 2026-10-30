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
 * 人像掩码来源。
 *
 * @author DreamLu
 */
@Getter
public enum IdPhotoMaskSource {

	/**
	 * {@code mica-ai-matting}（U²-Net {@code u2netp}，320×320 输入）——
	 * 显著性目标检测，发丝等细边缘比人像分割更精细，证件照默认用它。
	 */
	MATTING("通用抠图", "mica-ai-matting"),

	/**
	 * {@code mica-ai-humanseg}（PP-HumanSeg V2-Lite，192×192 输入）——
	 * 人像专用语义分割，画面里没有人时更保守。
	 */
	HUMANSEG("人像分割", "mica-ai-humanseg");

	/** 中文名 */
	private final String label;

	/** 能力模块名 */
	private final String module;

	IdPhotoMaskSource(String label, String module) {
		this.label = label;
		this.module = module;
	}

	/**
	 * 按名称解析。
	 *
	 * @param name 掩码来源名
	 * @return 对应枚举；无法解析时返回 null
	 */
	public static IdPhotoMaskSource of(String name) {
		if (name == null || name.trim().isEmpty()) {
			return null;
		}
		String key = name.trim();
		for (IdPhotoMaskSource source : values()) {
			if (source.name().equalsIgnoreCase(key) || source.label.equals(key)) {
				return source;
			}
		}
		return null;
	}

}
