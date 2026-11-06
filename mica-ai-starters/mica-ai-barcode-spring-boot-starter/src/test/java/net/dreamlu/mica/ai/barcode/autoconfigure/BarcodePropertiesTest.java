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

import net.dreamlu.mica.ai.barcode.config.BarcodeConfig;
import net.dreamlu.mica.ai.barcode.model.BarcodeSymbology;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 配置绑定测试（纯绑定，不启动 Spring 上下文、不加载模型）。
 * 本模块与 mica-ai-track 一样刻意零依赖，参数校验用 JDK 原生异常。
 */
class BarcodePropertiesTest {

	private static BarcodeProperties bind(Map<String, Object> source) {
		Binder binder = new Binder(new MapConfigurationPropertySource(source));
		return binder.bind("mica.ai.barcode", Bindable.of(BarcodeProperties.class))
			.orElseGet(BarcodeProperties::new);
	}

	private static Map<String, Object> allProperties() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.barcode.enabled", "false");
		src.put("mica.ai.barcode.formats", "QR_CODE,CODE_128,EAN_13");
		src.put("mica.ai.barcode.charset", "GBK");
		src.put("mica.ai.barcode.try-harder", "false");
		src.put("mica.ai.barcode.multi-scale", "false");
		src.put("mica.ai.barcode.upscale-factors", "2,4");
		src.put("mica.ai.barcode.max-side-length", "2048");
		src.put("mica.ai.barcode.tile-scan", "false");
		src.put("mica.ai.barcode.tile-rows", "3");
		src.put("mica.ai.barcode.tile-cols", "4");
		src.put("mica.ai.barcode.tile-overlap-ratio", "0.25");
		src.put("mica.ai.barcode.global-histogram-fallback", "false");
		src.put("mica.ai.barcode.max-results", "5");
		return src;
	}

	@Test
	void defaultsShouldMatchCoreConfigDefaults() {
		BarcodeProperties properties = new BarcodeProperties();
		BarcodeConfig config = BarcodeConfig.DEFAULTS;

		assertThat(properties.isEnabled()).isTrue();
		assertThat(properties.getFormats()).isEqualTo(config.getFormats())
			.isEqualTo(BarcodeSymbology.DEFAULT_FORMATS);
		assertThat(properties.getCharset()).isEqualTo(config.getCharset());
		assertThat(properties.isTryHarder()).isEqualTo(config.isTryHarder());
		assertThat(properties.isMultiScale()).isEqualTo(config.isMultiScale());
		assertThat(properties.getUpscaleFactors()).isEqualTo(config.getUpscaleFactors());
		assertThat(properties.getMaxSideLength()).isEqualTo(config.getMaxSideLength());
		assertThat(properties.isTileScan()).isEqualTo(config.isTileScan());
		assertThat(properties.getTileRows()).isEqualTo(config.getTileRows());
		assertThat(properties.getTileCols()).isEqualTo(config.getTileCols());
		assertThat(properties.getTileOverlapRatio()).isEqualTo(config.getTileOverlapRatio());
		assertThat(properties.isGlobalHistogramFallback()).isEqualTo(config.isGlobalHistogramFallback());
		assertThat(properties.getMaxResults()).isEqualTo(config.getMaxResults());
		assertThat(properties.toConfig().getFormats()).isEqualTo(BarcodeSymbology.DEFAULT_FORMATS);
		properties.toConfig().validate();
	}

	@Test
	void kebabCaseKeysShouldBind() {
		BarcodeProperties properties = bind(allProperties());

		assertThat(properties.isEnabled()).isFalse();
		assertThat(properties.getFormats()).containsExactly(
			BarcodeSymbology.QR_CODE, BarcodeSymbology.CODE_128, BarcodeSymbology.EAN_13);
		assertThat(properties.getCharset()).isEqualTo("GBK");
		assertThat(properties.isTryHarder()).isFalse();
		assertThat(properties.isMultiScale()).isFalse();
		assertThat(properties.getUpscaleFactors()).containsExactly(2, 4);
		assertThat(properties.getMaxSideLength()).isEqualTo(2048);
		assertThat(properties.isTileScan()).isFalse();
		assertThat(properties.getTileRows()).isEqualTo(3);
		assertThat(properties.getTileCols()).isEqualTo(4);
		assertThat(properties.getTileOverlapRatio()).isEqualTo(0.25f);
		assertThat(properties.isGlobalHistogramFallback()).isFalse();
		assertThat(properties.getMaxResults()).isEqualTo(5);
	}

	/**
	 * 装配一致性：toConfig 必须把每个属性都装进 BarcodeConfig，漏一项即失败。
	 */
	@Test
	void toConfigShouldWireEveryProperty() {
		BarcodeConfig config = bind(allProperties()).toConfig();

		assertThat(config.getFormats()).containsExactly(
			BarcodeSymbology.QR_CODE, BarcodeSymbology.CODE_128, BarcodeSymbology.EAN_13);
		assertThat(config.getCharset()).isEqualTo("GBK");
		assertThat(config.isTryHarder()).isFalse();
		assertThat(config.isMultiScale()).isFalse();
		assertThat(config.getUpscaleFactors()).containsExactly(2, 4);
		assertThat(config.getMaxSideLength()).isEqualTo(2048);
		assertThat(config.isTileScan()).isFalse();
		assertThat(config.getTileRows()).isEqualTo(3);
		assertThat(config.getTileCols()).isEqualTo(4);
		assertThat(config.getTileOverlapRatio()).isEqualTo(0.25f);
		assertThat(config.isGlobalHistogramFallback()).isFalse();
		assertThat(config.getMaxResults()).isEqualTo(5);
		config.validate();
	}

	@Test
	void illegalConfigShouldFailFast() {
		BarcodeProperties emptyFormats = new BarcodeProperties();
		emptyFormats.setFormats(Collections.<BarcodeSymbology>emptyList());
		assertThatThrownBy(() -> emptyFormats.toConfig().validate())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("formats");

		BarcodeProperties badOverlap = new BarcodeProperties();
		badOverlap.setTileOverlapRatio(0.8f);
		assertThatThrownBy(() -> badOverlap.toConfig().validate())
			.isInstanceOf(IllegalArgumentException.class)
			.hasMessageContaining("tileOverlapRatio");
	}
}
