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
package net.dreamlu.mica.ai.barcode;

import net.dreamlu.mica.ai.barcode.config.BarcodeConfig;
import net.dreamlu.mica.ai.barcode.model.BarcodeSymbology;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class BarcodeConfigTest {

	@Test
	void defaultsShouldMatchSharedDefaults() {
		BarcodeConfig config = BarcodeConfig.builder().build();

		assertThat(config.getFormats()).isEqualTo(BarcodeConfig.DEFAULTS.getFormats());
		assertThat(config.getFormats()).isEqualTo(BarcodeSymbology.DEFAULT_FORMATS);
		assertThat(config.getCharset()).isEqualTo("UTF-8");
		assertThat(config.isTryHarder()).isTrue();
		assertThat(config.isMultiScale()).isTrue();
		assertThat(config.getUpscaleFactors()).containsExactly(2, 3);
		assertThat(config.getMaxSideLength()).isEqualTo(4096);
		assertThat(config.isTileScan()).isTrue();
		assertThat(config.getTileRows()).isEqualTo(2);
		assertThat(config.getTileCols()).isEqualTo(2);
		assertThat(config.getTileOverlapRatio()).isEqualTo(0.12f);
		assertThat(config.isGlobalHistogramFallback()).isTrue();
		assertThat(config.getMaxResults()).isEqualTo(16);
		config.validate();
	}

	@Test
	void collectionsShouldBeReadOnlyViews() {
		BarcodeConfig config = BarcodeConfig.DEFAULTS;

		assertThatThrownBy(() -> config.getFormats().add(BarcodeSymbology.AZTEC))
			.isInstanceOf(UnsupportedOperationException.class);
		assertThatThrownBy(() -> config.getUpscaleFactors().add(4))
			.isInstanceOf(UnsupportedOperationException.class);
	}

	@Test
	void illegalConfigShouldFailFast() {
		assertThatThrownBy(() -> BarcodeConfig.builder().formats(Collections.<BarcodeSymbology>emptyList()).build().validate())
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("formats");
		assertThatThrownBy(() -> BarcodeConfig.builder().formats(Arrays.asList(BarcodeSymbology.QR_CODE, null)).build().validate())
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("null");
		assertThatThrownBy(() -> BarcodeConfig.builder().charset("NOT-A-CHARSET").build().validate())
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("charset");
		assertThatThrownBy(() -> BarcodeConfig.builder().upscaleFactors(Collections.singletonList(1)).build().validate())
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("upscaleFactors");
		assertThatThrownBy(() -> BarcodeConfig.builder().maxSideLength(0).build().validate())
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("maxSideLength");
		assertThatThrownBy(() -> BarcodeConfig.builder().tileCols(9).build().validate())
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("tileRows / tileCols");
		assertThatThrownBy(() -> BarcodeConfig.builder().tileOverlapRatio(0.6f).build().validate())
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("tileOverlapRatio");
		assertThatThrownBy(() -> BarcodeConfig.builder().maxResults(0).build().validate())
			.isInstanceOf(IllegalArgumentException.class).hasMessageContaining("maxResults");
	}
}
