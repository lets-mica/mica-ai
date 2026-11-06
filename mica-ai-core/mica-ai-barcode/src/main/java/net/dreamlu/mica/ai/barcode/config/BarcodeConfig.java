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
package net.dreamlu.mica.ai.barcode.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.barcode.model.BarcodeSymbology;

import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 条码解码配置。默认值全部集中在 {@link #DEFAULTS}（由 builder 生成），
 * Starter 的 Properties 直接引用它，避免「改了 core 默认值、YAML 忘了跟」。
 *
 * <p>本模块刻意零依赖（不引 {@code mica-ai-common}，避免把 ONNX Runtime 拖进纯算法模块），
 * 参数校验统一用 JDK 原生 {@link IllegalArgumentException}，与 {@code mica-ai-track} 一致。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class BarcodeConfig {

	public static final BarcodeConfig DEFAULTS = BarcodeConfig.builder().build();

	@Builder.Default
	private List<BarcodeSymbology> formats = BarcodeSymbology.DEFAULT_FORMATS;

	@Builder.Default
	private String charset = "UTF-8";

	@Builder.Default
	private boolean tryHarder = true;

	@Builder.Default
	private boolean multiScale = true;

	@Builder.Default
	private List<Integer> upscaleFactors = Collections.unmodifiableList(Arrays.asList(2, 3));

	/**
	 * 放大后长边上限（像素）。超过该值的放大倍数整档跳过，避免大图 ×3 直接 OOM。
	 */
	@Builder.Default
	private int maxSideLength = 4096;

	@Builder.Default
	private boolean tileScan = true;

	@Builder.Default
	private int tileRows = 2;

	@Builder.Default
	private int tileCols = 2;

	@Builder.Default
	private float tileOverlapRatio = 0.12f;

	@Builder.Default
	private boolean globalHistogramFallback = true;

	@Builder.Default
	private int maxResults = 16;

	public List<BarcodeSymbology> getFormats() {
		return formats == null ? Collections.<BarcodeSymbology>emptyList()
			: Collections.unmodifiableList(formats);
	}

	public List<Integer> getUpscaleFactors() {
		return upscaleFactors == null ? Collections.<Integer>emptyList()
			: Collections.unmodifiableList(upscaleFactors);
	}

	public void validate() {
		List<BarcodeSymbology> currentFormats = formats;
		if (currentFormats == null || currentFormats.isEmpty()) {
			throw new IllegalArgumentException("formats 不能为空：至少要启用一种条码制式");
		}
		for (BarcodeSymbology format : currentFormats) {
			if (format == null) {
				throw new IllegalArgumentException("formats 不能包含 null");
			}
		}
		if (charset == null || charset.trim().isEmpty() || !Charset.isSupported(charset)) {
			throw new IllegalArgumentException("charset 必须是受支持的字符集名称: " + charset);
		}
		List<Integer> factors = upscaleFactors;
		if (multiScale) {
			if (factors == null || factors.isEmpty()) {
				throw new IllegalArgumentException("multiScale=true 时 upscaleFactors 不能为空");
			}
			for (Integer factor : factors) {
				if (factor == null || factor < 2 || factor > 8) {
					throw new IllegalArgumentException("upscaleFactors 每项必须在 [2,8]: " + factor);
				}
			}
		}
		if (maxSideLength <= 0) {
			throw new IllegalArgumentException("maxSideLength 必须为正数");
		}
		if (tileScan && (tileRows < 1 || tileRows > 4 || tileCols < 1 || tileCols > 4)) {
			throw new IllegalArgumentException("tileRows / tileCols 必须在 [1,4]");
		}
		if (tileOverlapRatio < 0f || tileOverlapRatio >= 0.5f) {
			throw new IllegalArgumentException("tileOverlapRatio 必须在 [0, 0.5)");
		}
		if (maxResults <= 0) {
			throw new IllegalArgumentException("maxResults 必须为正数");
		}
	}
}
