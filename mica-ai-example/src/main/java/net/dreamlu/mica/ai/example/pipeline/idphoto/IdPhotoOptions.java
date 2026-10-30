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

import lombok.Builder;
import lombok.Getter;

/**
 * 证件照一键生成的参数。
 *
 * @author DreamLu
 */
@Getter
@Builder(toBuilder = true)
public class IdPhotoOptions {

	/** 尺寸规格，默认一寸 */
	@Builder.Default
	private final IdPhotoSpec spec = IdPhotoSpec.ONE;

	/** 底色，默认白底 */
	@Builder.Default
	private final IdPhotoBackground background = IdPhotoBackground.WHITE;

	/** 目标分辨率 DPI，默认 300（印刷级） */
	@Builder.Default
	private final int dpi = 300;

	/** 人像掩码来源，默认通用抠图 */
	@Builder.Default
	private final IdPhotoMaskSource maskSource = IdPhotoMaskSource.MATTING;

	/**
	 * 是否做质量门控（人脸质量分低于门限直接判不合格）。
	 *
	 * <p>默认 false：质量评估照做、分数照回填，但<b>不拦截</b>出片——
	 * 「拍糊了」由用户自己看着分数决定要不要重拍，而不是由流程替他拒收。
	 * 需要严格把关时置 true，质量不达标会直接返回 LOW_QUALITY 且不产出成片。
	 */
	@Builder.Default
	private final boolean qualityGate = false;

	/** 质量门限，与 {@code mica.ai.face.quality.threshold} 语义一致 */
	@Builder.Default
	private final float minQuality = 0.5f;

	/** 是否对成片超分（输出 4 倍尺寸，供印刷） */
	@Builder.Default
	private final boolean upscale = true;

	/**
	 * 头顶到成片上边缘的留白比例。
	 *
	 * <p>证件照通用规范要求头顶留白 5%~10%，取 7% 居中。
	 */
	@Builder.Default
	private final float topMarginRatio = 0.07f;

	/**
	 * 头部高度占成片高度的比例。
	 *
	 * <p>证件照通用规范要求头部占比 60%~75%，取 75%。
	 * 与 {@link #topMarginRatio}（7%）配合后，下巴留白落在 18%，同样在规范的 15%~20% 内。
	 */
	@Builder.Default
	private final float headHeightRatio = 0.75f;

	/**
	 * 脸高到头高的倍数（含头发与下巴余量）。
	 */
	@Builder.Default
	private final float headToFaceRatio = 2.0f;

}
