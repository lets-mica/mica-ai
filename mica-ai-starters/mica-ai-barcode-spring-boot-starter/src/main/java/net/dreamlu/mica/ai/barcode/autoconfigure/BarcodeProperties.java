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
package net.dreamlu.mica.ai.barcode.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.barcode.config.BarcodeConfig;
import net.dreamlu.mica.ai.barcode.model.BarcodeSymbology;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.ArrayList;
import java.util.List;

/**
 * mica-ai-barcode 配置属性，对应 {@code mica.ai.barcode} 前缀。
 *
 * <p>默认值一律从 {@link BarcodeConfig#DEFAULTS} 读取，避免 core 与 YAML 各写一份默认值。
 * 本能力没有必填项（无模型路径），不配任何 YAML 也能直接装配出可用引擎。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.barcode")
public class BarcodeProperties {

	private boolean enabled = true;

	private List<BarcodeSymbology> formats = new ArrayList<>(BarcodeConfig.DEFAULTS.getFormats());

	private String charset = BarcodeConfig.DEFAULTS.getCharset();

	private boolean tryHarder = BarcodeConfig.DEFAULTS.isTryHarder();

	private boolean multiScale = BarcodeConfig.DEFAULTS.isMultiScale();

	private List<Integer> upscaleFactors = new ArrayList<>(BarcodeConfig.DEFAULTS.getUpscaleFactors());

	private int maxSideLength = BarcodeConfig.DEFAULTS.getMaxSideLength();

	private boolean tileScan = BarcodeConfig.DEFAULTS.isTileScan();

	private int tileRows = BarcodeConfig.DEFAULTS.getTileRows();

	private int tileCols = BarcodeConfig.DEFAULTS.getTileCols();

	private float tileOverlapRatio = BarcodeConfig.DEFAULTS.getTileOverlapRatio();

	private boolean globalHistogramFallback = BarcodeConfig.DEFAULTS.isGlobalHistogramFallback();

	private int maxResults = BarcodeConfig.DEFAULTS.getMaxResults();

	public BarcodeConfig toConfig() {
		return BarcodeConfig.builder()
			.formats(formats)
			.charset(charset)
			.tryHarder(tryHarder)
			.multiScale(multiScale)
			.upscaleFactors(upscaleFactors)
			.maxSideLength(maxSideLength)
			.tileScan(tileScan)
			.tileRows(tileRows)
			.tileCols(tileCols)
			.tileOverlapRatio(tileOverlapRatio)
			.globalHistogramFallback(globalHistogramFallback)
			.maxResults(maxResults)
			.build();
	}
}
