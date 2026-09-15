/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype.postprocess;

import net.dreamlu.mica.ai.filetype.PredictionMode;
import net.dreamlu.mica.ai.filetype.config.ContentTypeRegistry;
import net.dreamlu.mica.ai.filetype.config.ModelConfig;
import net.dreamlu.mica.ai.filetype.model.ContentTypeInfo;
import net.dreamlu.mica.ai.filetype.model.ContentTypeLabel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

class PredictionPostProcessorTest {

	private ModelConfig modelConfig;
	private ContentTypeRegistry registry;

	@BeforeEach
	void setUp() {
		modelConfig = new ModelConfig();
		modelConfig.setBegSize(1024);
		modelConfig.setEndSize(1024);
		modelConfig.setMidSize(0);
		modelConfig.setPaddingToken(256);
		modelConfig.setBlockSize(4096);
		modelConfig.setMinFileSizeForDl(8);
		modelConfig.setMediumConfidenceThreshold(0.5f);

		Map<String, Float> thresholds = new HashMap<>();
		thresholds.put("png", 0.99f);
		thresholds.put("txt", 0.99f);
		modelConfig.setThresholds(thresholds);

		Map<String, ContentTypeInfo> types = new HashMap<>();
		ContentTypeInfo png = new ContentTypeInfo();
		png.setLabel("png");
		png.setText(false);
		png.setMimeType("image/png");
		png.setGroup("image");
		types.put("png", png);

		ContentTypeInfo txt = new ContentTypeInfo();
		txt.setLabel("txt");
		txt.setText(true);
		txt.setMimeType("text/plain");
		txt.setGroup("text");
		types.put("txt", txt);

		ContentTypeInfo unknown = new ContentTypeInfo();
		unknown.setLabel(ContentTypeLabel.UNKNOWN);
		unknown.setText(false);
		unknown.setMimeType("application/octet-stream");
		unknown.setGroup("unknown");
		types.put(ContentTypeLabel.UNKNOWN, unknown);

		registry = new ContentTypeRegistry(types);
	}

	@Test
	void bestGuess_alwaysReturnsDlLabel() {
		PredictionPostProcessor p = new PredictionPostProcessor(modelConfig, registry, PredictionMode.BEST_GUESS);
		assertThat(p.resolveOutputLabel("png", 0.1f)).isEqualTo("png");
		assertThat(p.resolveOutputLabel("png", 0.0f)).isEqualTo("png");
	}

	@Test
	void mediumConfidence_belowThreshold_unknown_fallback() {
		PredictionPostProcessor p = new PredictionPostProcessor(modelConfig, registry, PredictionMode.MEDIUM_CONFIDENCE);
		assertThat(p.resolveOutputLabel("png", 0.3f)).isEqualTo(ContentTypeLabel.UNKNOWN);
	}

	@Test
	void mediumConfidence_aboveThreshold_returnsDlLabel() {
		PredictionPostProcessor p = new PredictionPostProcessor(modelConfig, registry, PredictionMode.MEDIUM_CONFIDENCE);
		assertThat(p.resolveOutputLabel("png", 0.9f)).isEqualTo("png");
	}

	@Test
	void highConfidence_usesLabelSpecificThreshold() {
		PredictionPostProcessor p = new PredictionPostProcessor(modelConfig, registry, PredictionMode.HIGH_CONFIDENCE);
		assertThat(p.resolveOutputLabel("png", 0.5f)).isEqualTo(ContentTypeLabel.UNKNOWN);
		assertThat(p.resolveOutputLabel("png", 0.995f)).isEqualTo("png");
	}

	@Test
	void highConfidence_fallsBackToMediumWhenLabelMissingInThresholds() {
		Map<String, Float> onlyPng = new HashMap<>();
		onlyPng.put("png", 0.99f);
		modelConfig.setThresholds(onlyPng);

		PredictionPostProcessor p = new PredictionPostProcessor(modelConfig, registry, PredictionMode.HIGH_CONFIDENCE);
		assertThat(p.resolveOutputLabel("txt", 0.5f)).isEqualTo("txt");
		assertThat(p.resolveOutputLabel("txt", 0.3f)).isEqualTo(ContentTypeLabel.TXT);
	}

	@Test
	void mediumConfidence_textFallbackReturnsTxt() {
		PredictionPostProcessor p = new PredictionPostProcessor(modelConfig, registry, PredictionMode.MEDIUM_CONFIDENCE);
		Map<String, ContentTypeInfo> textTypes = new HashMap<>();
		ContentTypeInfo src = new ContentTypeInfo();
		src.setLabel("src");
		src.setText(true);
		textTypes.put("src", src);
		ContentTypeRegistry textRegistry = new ContentTypeRegistry(textTypes);

		PredictionPostProcessor textP = new PredictionPostProcessor(modelConfig, textRegistry, PredictionMode.MEDIUM_CONFIDENCE);
		assertThat(textP.resolveOutputLabel("src", 0.1f)).isEqualTo(ContentTypeLabel.TXT);
	}

	@Test
	void overwriteMap_appliedBeforeThresholdCheck() {
		Map<String, String> overwrite = new HashMap<>();
		overwrite.put("python", "txt");
		modelConfig.setOverwriteMap(overwrite);

		PredictionPostProcessor p = new PredictionPostProcessor(modelConfig, registry, PredictionMode.BEST_GUESS);
		assertThat(p.resolveOutputLabel("python", 0.1f)).isEqualTo("txt");
	}

	@Test
	void overwriteMap_doesNotBypassLowConfidence() {
		Map<String, String> overwrite = new HashMap<>();
		overwrite.put("python", "txt");
		modelConfig.setOverwriteMap(overwrite);

		PredictionPostProcessor p = new PredictionPostProcessor(modelConfig, registry, PredictionMode.MEDIUM_CONFIDENCE);
		assertThat(p.resolveOutputLabel("python", 0.1f)).isEqualTo(ContentTypeLabel.TXT);
	}

	@Test
	void defaultMediumThreshold_is05() {
		assertThat(modelConfig.getMediumConfidenceThreshold()).isEqualTo(0.5f);
	}
}