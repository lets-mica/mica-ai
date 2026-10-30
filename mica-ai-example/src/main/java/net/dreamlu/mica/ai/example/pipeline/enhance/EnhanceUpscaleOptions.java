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

import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;

/**
 * 图像超分的可调参数。
 *
 * <p>模型（Real-ESRGAN {@code realesr-general-x4v3}）原生只产出 4 倍尺寸，
 * 这里把「输出倍率」开放成可调项：倍率小于原生值时先按 4 倍推理、再重采样回目标倍率，
 * 得到的边缘比直接双三次插值更锐（先重建高频、再降采样，比直接插值少一次细节丢失）。
 *
 * @author DreamLu
 */
@Getter
@Builder(toBuilder = true)
public class EnhanceUpscaleOptions {

	/** 输出倍率（相对输入尺寸），取值 1 ~ 模型原生倍率 */
	@Builder.Default
	private final int outScale = 4;

	/**
	 * 分块推理的 tile 边长（原图像素）。
	 *
	 * <p>越小越省内存、越慢；越大越快、峰值内存越高。
	 */
	@Builder.Default
	private final int tileSize = 256;

	/**
	 * 输入长边上限（像素）。
	 *
	 * <p>超过上限先等比缩小再超分，避免超大图把内存和响应体打爆。
	 */
	@Builder.Default
	private final int maxInputSide = 1200;

	/**
	 * 校验参数合法性。
	 *
	 * @param modelScale 模型原生倍率
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 参数越界
	 */
	public void validate(int modelScale) {
		if (outScale < 1 || outScale > modelScale) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"输出倍率 outScale 必须在 1 ~ " + modelScale + " 之间，当前: " + outScale);
		}
		if (tileSize < 32 || tileSize > 2048) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"分块边长 tileSize 必须在 32 ~ 2048 之间，当前: " + tileSize);
		}
		if (maxInputSide < 64) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"输入长边上限 maxInputSide 不能小于 64，当前: " + maxInputSide);
		}
	}

}
