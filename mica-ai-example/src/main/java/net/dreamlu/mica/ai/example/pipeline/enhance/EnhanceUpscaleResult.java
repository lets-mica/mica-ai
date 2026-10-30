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

/**
 * 图像超分结果。
 *
 * <p>{@code originalPng} 是<b>实际送进模型</b>的那张图（若触发了输入长边上限，
 * 就是缩小后的版本），{@code upscaledPng} 是超分输出。两者同源同尺寸基准，
 * 可以直接拿来做逐像素对比，不会出现「拿大图比小图」的错觉。
 *
 * @author DreamLu
 */
@Getter
public class EnhanceUpscaleResult {

	/** 送入模型的原图 PNG 字节 */
	private final byte[] originalPng;

	/** 送入模型的原图宽度 */
	private final int originalWidth;

	/** 送入模型的原图高度 */
	private final int originalHeight;

	/** 超分输出 PNG 字节 */
	private final byte[] upscaledPng;

	/** 超分输出宽度 */
	private final int upscaledWidth;

	/** 超分输出高度 */
	private final int upscaledHeight;

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
	private final String modelVersion;

	/**
	 * 构造超分结果。
	 *
	 * @param originalPng 送入模型的原图 PNG
	 * @param originalWidth 送入模型的原图宽
	 * @param originalHeight 送入模型的原图高
	 * @param upscaledPng 超分输出 PNG
	 * @param upscaledWidth 输出宽
	 * @param upscaledHeight 输出高
	 * @param sourceWidth 上传原图宽
	 * @param sourceHeight 上传原图高
	 * @param preDownscaled 是否先缩小过
	 * @param outScale 输出倍率
	 * @param modelScale 模型原生倍率
	 * @param tileSize 分块边长
	 * @param elapsedMs 耗时（毫秒）
	 * @param modelVersion 模型标识
	 */
	public EnhanceUpscaleResult(byte[] originalPng, int originalWidth, int originalHeight,
		byte[] upscaledPng, int upscaledWidth, int upscaledHeight,
		int sourceWidth, int sourceHeight, boolean preDownscaled,
		int outScale, int modelScale, int tileSize, long elapsedMs, String modelVersion) {
		this.originalPng = originalPng;
		this.originalWidth = originalWidth;
		this.originalHeight = originalHeight;
		this.upscaledPng = upscaledPng;
		this.upscaledWidth = upscaledWidth;
		this.upscaledHeight = upscaledHeight;
		this.sourceWidth = sourceWidth;
		this.sourceHeight = sourceHeight;
		this.preDownscaled = preDownscaled;
		this.outScale = outScale;
		this.modelScale = modelScale;
		this.tileSize = tileSize;
		this.elapsedMs = elapsedMs;
		this.modelVersion = modelVersion;
	}

}
