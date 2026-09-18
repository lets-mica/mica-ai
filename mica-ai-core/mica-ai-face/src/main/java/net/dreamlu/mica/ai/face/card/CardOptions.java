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
package net.dreamlu.mica.ai.face.card;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;

@Data
@Builder(toBuilder = true)
@NoArgsConstructor
@AllArgsConstructor
public class CardOptions {

	@Builder.Default
	private int outputWidth = 1011;

	@Builder.Default
	private int outputHeight = 638;

	@Builder.Default
	private double aspectTolerance = 0.35;

	@Builder.Default
	private double minAreaRatio = 0.002;

	@Builder.Default
	private int borderMargin = 3;

	@Builder.Default
	private int detectMaxSide = 1100;

	@Builder.Default
	private int saturationCeiling = 110;

	@Builder.Default
	private int closeKernelSize = 13;

	@Builder.Default
	private boolean autoOrient = true;

	@Builder.Default
	private int rotationDegrees = 0;

	@Builder.Default
	private boolean enhance = true;

	@Builder.Default
	private double sharpenAmount = 2.2;

	@Builder.Default
	private double sharpenSigma = 1.6;

	@Builder.Default
	private double claheClip = 1.6;

	@Builder.Default
	private int minCardSize = 400;

	public static CardOptions defaults() {
		return CardOptions.builder().build();
	}

	public static CardOptions of(int width, int height) {
		return CardOptions.builder().outputWidth(width).outputHeight(height).build();
	}

	public static CardOptions passport() {
		return CardOptions.builder().outputWidth(1476).outputHeight(1039).build();
	}

	public double expectedAspectRatio() {
		double w = Math.max(outputWidth, outputHeight);
		double h = Math.min(outputWidth, outputHeight);
		return w / h;
	}

	void validate() {
		if (outputWidth < 16 || outputWidth > 8192) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"outputWidth 需在 [16, 8192]，实际: " + outputWidth);
		}
		if (outputHeight < 16 || outputHeight > 8192) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"outputHeight 需在 [16, 8192]，实际: " + outputHeight);
		}
		if (!(aspectTolerance > 0.0 && aspectTolerance <= 1.0)) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"aspectTolerance 需在 (0, 1]，实际: " + aspectTolerance);
		}
		if (!(minAreaRatio > 0.0 && minAreaRatio < 1.0)) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"minAreaRatio 需在 (0, 1)，实际: " + minAreaRatio);
		}
		if (borderMargin < 0) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED, "borderMargin 不能为负");
		}
		if (detectMaxSide != 0 && detectMaxSide < 64) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"detectMaxSide 需为 0（不缩放）或不小于 64，实际: " + detectMaxSide);
		}
		if (saturationCeiling < 0 || saturationCeiling > 255) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"saturationCeiling 需在 [0, 255]，实际: " + saturationCeiling);
		}
		if (closeKernelSize < 3 || closeKernelSize > 199 || closeKernelSize % 2 == 0) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"closeKernelSize 需为 [3, 199] 内的奇数，实际: " + closeKernelSize);
		}
		if (rotationDegrees % 90 != 0) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"rotationDegrees 需为 90 的整数倍，实际: " + rotationDegrees);
		}
		if (!(sharpenAmount >= 1.0 && sharpenAmount <= 5.0)) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"sharpenAmount 需在 [1.0, 5.0]，实际: " + sharpenAmount);
		}
		if (!(sharpenSigma > 0.0 && sharpenSigma <= 10.0)) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"sharpenSigma 需在 (0, 10]，实际: " + sharpenSigma);
		}
		if (!(claheClip >= 0.0 && claheClip <= 40.0)) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"claheClip 需在 [0, 40]，实际: " + claheClip);
		}
		if (minCardSize < 0) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED, "minCardSize 不能为负");
		}
	}
}