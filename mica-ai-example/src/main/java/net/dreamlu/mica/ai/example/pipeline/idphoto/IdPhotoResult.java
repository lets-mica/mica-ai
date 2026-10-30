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
 * 证件照一键生成的结果。
 *
 * <p>开启质量门控且质量不达标时 {@link #isPassed()} 为 false、{@link #getOutputPng()} 为 null，
 * 但人脸框与质量分仍然回填，方便调用方告诉用户「哪里不行」；
 * 门控关闭时质量分照常评估回填，只是不拦截出片。
 *
 * @author DreamLu
 */
@Getter
public class IdPhotoResult {

	/** 判定通过 */
	private final boolean passed;

	/** 结果码：OK / NO_FACE / LOW_QUALITY / MASK_FAILED */
	private final String reason;

	/** 给人看的提示 */
	private final String hint;

	/** 成片 PNG 字节；未产出时为 null */
	private final byte[] outputPng;

	/** 成片宽 */
	private final int width;

	/** 成片高 */
	private final int height;

	/** 人脸质量分（eDifFIQA），未做人脸质量评估时为 -1 */
	private final float qualityScore;

	/**
	 * 人脸质量是否达标（质量分 &ge; 门限）。
	 *
	 * <p>三态：{@code null} 表示未做人脸质量评估（质量模型未启用），
	 * {@code true}/{@code false} 表示评估过且达标 / 不达标。
	 * 不达标只在开启质量门控时才会拦住出片，否则仅作为参考回填。
	 */
	private final Boolean qualityPassed;

	/** 人脸检测置信度 */
	private final float faceScore;

	/** 人脸框 [x1, y1, x2, y2]（原图坐标） */
	private final int[] faceBox;

	/** 尺寸规格名 */
	private final String spec;

	/** 底色名 */
	private final String background;

	/** 掩码来源名 */
	private final String maskSource;

	/** 源图上的裁剪矩形 [x1, y1, x2, y2]，越界部分由底色补齐 */
	private final int[] cropBox;

	/** 是否已超分 */
	private final boolean upscaled;

	/** 超分倍数，未超分为 1 */
	private final int upscaleScale;

	public IdPhotoResult(boolean passed, String reason, String hint,
						 byte[] outputPng, int width, int height,
						 float qualityScore, Boolean qualityPassed, float faceScore, int[] faceBox,
						 String spec, String background, String maskSource,
						 int[] cropBox, boolean upscaled, int upscaleScale) {
		this.passed = passed;
		this.reason = reason;
		this.hint = hint;
		this.outputPng = outputPng;
		this.width = width;
		this.height = height;
		this.qualityScore = qualityScore;
		this.qualityPassed = qualityPassed;
		this.faceScore = faceScore;
		this.faceBox = faceBox;
		this.spec = spec;
		this.background = background;
		this.maskSource = maskSource;
		this.cropBox = cropBox;
		this.upscaled = upscaled;
		this.upscaleScale = upscaleScale;
	}

}
