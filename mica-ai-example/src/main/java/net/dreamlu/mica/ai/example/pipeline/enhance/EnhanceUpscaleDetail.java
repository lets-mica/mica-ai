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
package net.dreamlu.mica.ai.example.pipeline.enhance;

import lombok.Getter;

import java.util.Base64;

/**
 * 图像超分详情响应体。
 *
 * <p>同时带上「送入模型的原图」与「超分输出」两张图（base64 内联），
 * 供前端做同源对比滑块；再附上尺寸、倍率、耗时等元信息。
 *
 * @author DreamLu
 */
@Getter
public class EnhanceUpscaleDetail {

	/** 送入模型的原图宽度 */
	private final int originalWidth;

	/** 送入模型的原图高度 */
	private final int originalHeight;

	/** 送入模型的原图（data URI，PNG） */
	private final String originalBase64;

	/** 超分输出宽度 */
	private final int outputWidth;

	/** 超分输出高度 */
	private final int outputHeight;

	/** 超分输出（data URI，PNG） */
	private final String outputBase64;

	/** 上传文件的原始宽度（未经输入长边上限处理） */
	private final int sourceWidth;

	/** 上传文件的原始高度（未经输入长边上限处理） */
	private final int sourceHeight;

	/** 是否因超过输入长边上限而先缩小过 */
	private final boolean preDownscaled;

	/** 实际输出倍率 */
	private final int outScale;

	/** 模型原生倍率 */
	private final int modelScale;

	/** 实际使用的分块边长 */
	private final int tileSize;

	/** 端到端耗时（毫秒） */
	private final long elapsedMs;

	/** 模型标识 */
	private final String model;

	/**
	 * 由超分结果构造响应体。
	 *
	 * @param result 超分结果
	 */
	public EnhanceUpscaleDetail(EnhanceUpscaleResult result) {
		this.originalWidth = result.getOriginalWidth();
		this.originalHeight = result.getOriginalHeight();
		this.originalBase64 = toDataUri(result.getOriginalPng());
		this.outputWidth = result.getUpscaledWidth();
		this.outputHeight = result.getUpscaledHeight();
		this.outputBase64 = toDataUri(result.getUpscaledPng());
		this.sourceWidth = result.getSourceWidth();
		this.sourceHeight = result.getSourceHeight();
		this.preDownscaled = result.isPreDownscaled();
		this.outScale = result.getOutScale();
		this.modelScale = result.getModelScale();
		this.tileSize = result.getTileSize();
		this.elapsedMs = result.getElapsedMs();
		this.model = result.getModelVersion();
	}

	private static String toDataUri(byte[] png) {
		return "data:image/png;base64," + Base64.getEncoder().encodeToString(png);
	}

}
