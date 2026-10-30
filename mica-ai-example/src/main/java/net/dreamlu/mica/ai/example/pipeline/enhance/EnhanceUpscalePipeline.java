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

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.enhance.util.EnhanceImageUtils;
import net.dreamlu.mica.ai.example.config.EnhanceEnginePool;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.Objects;

/**
 * 图像超分流水线。
 *
 * <p>步骤：解码 →（超长边先等比缩小）→ 按档位取引擎 → 模型原生倍率推理 →
 * （输出倍率 ≠ 原生倍率时重采样）→ 编码 PNG。
 *
 * <p>关于「先重建高频再降采样」：输出倍率小于模型原生倍率时（4× → 2×），
 * 本实现是先跑完 4× 推理再面积平均降采样，而不是直接双三次插值到 2×，
 * 因为前者是在「已经重建出高频细节」的图上做降采样，边缘更实。
 *
 * @author DreamLu
 */
@Slf4j
@Component
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.enhance", name = "enabled", havingValue = "true", matchIfMissing = true)
public class EnhanceUpscalePipeline {

	private final EnhanceEnginePool enginePool;

	/**
	 * 模型原生倍率。
	 *
	 * @return 模型固定的输出倍率
	 */
	public int modelScale() {
		return enginePool.modelScale();
	}

	/**
	 * 执行超分。
	 *
	 * @param imageBytes 上传图片字节
	 * @param options 超分参数
	 * @return 超分结果（含对比用的原图与输出图 PNG）
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 参数非法
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码或推理失败
	 */
	public EnhanceUpscaleResult upscale(byte[] imageBytes, EnhanceUpscaleOptions options) {
		Objects.requireNonNull(imageBytes, "imageBytes must not be null");
		Objects.requireNonNull(options, "options must not be null");
		final long start = System.currentTimeMillis();
		Mat src = null;
		Mat preScaled = null;
		Mat upscaled = null;
		Mat outScaled = null;
		try {
			src = EnhanceImageUtils.byteArrayToMat(imageBytes);
			final int sourceWidth = src.cols();
			final int sourceHeight = src.rows();
			final int modelScale = enginePool.modelScale();
			options.validate(modelScale);

			Mat modelInput = src;
			boolean preDownscaled = false;
			final int longSide = Math.max(sourceWidth, sourceHeight);
			if (longSide > options.getMaxInputSide()) {
				final double ratio = (double) options.getMaxInputSide() / longSide;
				final int w = Math.max(1, (int) Math.round(sourceWidth * ratio));
				final int h = Math.max(1, (int) Math.round(sourceHeight * ratio));
				preScaled = new Mat();
				Imgproc.resize(src, preScaled, new Size(w, h), 0, 0, Imgproc.INTER_AREA);
				modelInput = preScaled;
				preDownscaled = true;
				log.info("输入长边 {} 超过上限 {}，先缩放到 {}x{} 再超分", longSide, options.getMaxInputSide(), w, h);
			}

			final EnhanceEngine engine = enginePool.engine(options.getTileSize());
			upscaled = engine.upscale(modelInput);
			if (upscaled == null || upscaled.empty()) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "超分推理结果为空");
			}

			Mat output = upscaled;
			if (options.getOutScale() != modelScale) {
				final double ratio = (double) options.getOutScale() / modelScale;
				final int w = Math.max(1, (int) Math.round(upscaled.cols() * ratio));
				final int h = Math.max(1, (int) Math.round(upscaled.rows() * ratio));
				outScaled = new Mat();
				Imgproc.resize(upscaled, outScaled, new Size(w, h), 0, 0, Imgproc.INTER_AREA);
				output = outScaled;
			}

			final byte[] originalPng = EnhanceImageUtils.imencodePng(modelInput);
			final byte[] upscaledPng = EnhanceImageUtils.imencodePng(output);
			return new EnhanceUpscaleResult(originalPng, modelInput.cols(), modelInput.rows(),
				upscaledPng, output.cols(), output.rows(),
				sourceWidth, sourceHeight, preDownscaled,
				options.getOutScale(), modelScale, options.getTileSize(),
				System.currentTimeMillis() - start, enginePool.modelVersion());
		} finally {
			EnhanceImageUtils.releaseAll(outScaled, upscaled, preScaled, src);
		}
	}

}
