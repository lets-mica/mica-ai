/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype.feature;

import net.dreamlu.mica.ai.filetype.config.ModelConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

class FeaturesExtractorTest {

	private static ModelConfig stdV3Config() {
		ModelConfig cfg = new ModelConfig();
		cfg.setBegSize(1024);
		cfg.setEndSize(1024);
		cfg.setMidSize(0);
		cfg.setPaddingToken(256);
		cfg.setBlockSize(4096);
		cfg.setMinFileSizeForDl(8);
		cfg.setMediumConfidenceThreshold(0.5f);
		return cfg;
	}

	@Test
	void beg_leftAligned_afterLstrip() {
		ModelConfig cfg = stdV3Config();
		byte[] head = new byte[8];
		head[0] = ' ';
		head[1] = '\t';
		head[2] = '\n';
		head[3] = 0x41; // 'A'
		head[4] = 0x42; // 'B'
		head[5] = 0x43; // 'C'
		head[6] = '\r';
		head[7] = 0x44; // 'D'

		int[] features = FeaturesExtractor.extract(cfg, head, head);

		assertThat(features[0]).isEqualTo(0x41);
		assertThat(features[1]).isEqualTo(0x42);
		assertThat(features[2]).isEqualTo(0x43);
		assertThat(features[3]).isEqualTo(0x0D);
		assertThat(features[4]).isEqualTo(0x44);
		assertThat(features[5]).isEqualTo(256);
		assertThat(features[cfg.featuresSize() - 1]).isEqualTo(0x44);
		assertThat(features[cfg.featuresSize() - 2]).isEqualTo(0x0D);
		assertThat(features[cfg.featuresSize() - 3]).isEqualTo(0x43);
		assertThat(features[cfg.featuresSize() - 4]).isEqualTo(0x42);
		assertThat(features[cfg.featuresSize() - 5]).isEqualTo(0x41);
	}

	@Test
	void end_rightAligned_afterRstrip() {
		ModelConfig cfg = stdV3Config();
		byte[] tail = new byte[8];
		tail[0] = 0x41;
		tail[1] = 0x42;
		tail[2] = '\n';
		tail[3] = '\n';
		tail[4] = '\n';
		tail[5] = '\n';
		tail[6] = '\n';
		tail[7] = '\n';

		int[] features = FeaturesExtractor.extract(cfg, tail, tail);

		assertThat(features[cfg.featuresSize() - 1]).isEqualTo(0x42);
		assertThat(features[cfg.featuresSize() - 2]).isEqualTo(0x41);
		assertThat(features[cfg.featuresSize() - 3]).isEqualTo(256);
	}

	@Test
	void padding_fillsUnwrittenSlots() {
		ModelConfig cfg = stdV3Config();
		byte[] empty = new byte[0];
		int[] features = FeaturesExtractor.extract(cfg, empty, empty);

		for (int f : features) {
			assertThat(f).isEqualTo(256);
		}
	}

	@Test
	void verticalTab_0x0B_isLstripWhitespace() {
		ModelConfig cfg = stdV3Config();
		byte[] data = new byte[5];
		data[0] = 0x0B;
		data[1] = 0x0B;
		data[2] = 0x41;
		data[3] = 0x42;
		data[4] = 0x43;

		int[] features = FeaturesExtractor.extract(cfg, data, data);

		assertThat(features[0]).isEqualTo(0x41);
		assertThat(features[1]).isEqualTo(0x42);
		assertThat(features[2]).isEqualTo(0x43);
		assertThat(features[3]).isEqualTo(256);
	}

	@Test
	void shortInput_stillTruncatedNotPaddedAtFront() {
		ModelConfig cfg = stdV3Config();
		byte[] data = new byte[]{0x41, 0x42, 0x43};

		int[] features = FeaturesExtractor.extract(cfg, data, data);

		assertThat(features[0]).isEqualTo(0x41);
		assertThat(features[1]).isEqualTo(0x42);
		assertThat(features[2]).isEqualTo(0x43);
		assertThat(features[3]).isEqualTo(256);
		assertThat(features[cfg.featuresSize() - 1]).isEqualTo(0x43);
		assertThat(features[cfg.featuresSize() - 2]).isEqualTo(0x42);
		assertThat(features[cfg.featuresSize() - 3]).isEqualTo(0x41);
	}

	@Test
	void hasEnoughMeaningfulBytes_respectsMinFileSizeForDl() {
		ModelConfig cfg = stdV3Config();
		cfg.setMinFileSizeForDl(8);

		int[] allPadding = new int[cfg.featuresSize()];
		java.util.Arrays.fill(allPadding, 256);
		assertThat(FeaturesExtractor.hasEnoughMeaningfulBytes(cfg, allPadding)).isFalse();

		int[] partlyFilled = new int[cfg.featuresSize()];
		java.util.Arrays.fill(partlyFilled, 256);
		partlyFilled[7] = 0x41;
		assertThat(FeaturesExtractor.hasEnoughMeaningfulBytes(cfg, partlyFilled)).isTrue();
	}
}