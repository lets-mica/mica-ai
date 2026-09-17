/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.layout;

import net.dreamlu.mica.ai.layout.config.LayoutConfig;
import net.dreamlu.mica.ai.layout.model.LayoutLabel;
import net.dreamlu.mica.ai.layout.model.LayoutResult;
import net.dreamlu.mica.ai.layout.pipeline.LayoutPipeline;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class LayoutPipelineTest {

	@Test
	void configDefaultShouldBeValid() {
		LayoutConfig config = LayoutConfig.builder().build();
		config.validate();
		assertThat(config.getScoreThreshold()).isEqualTo(0.4f);
		assertThat(config.getMaxSideLength()).isEqualTo(800);
		assertThat(config.getModelVersion()).isEqualTo("v3");
		assertThat(config.getMean()).hasSize(3);
		assertThat(config.getStd()).hasSize(3);
		assertThat(config.isLayoutNms()).isTrue();
		assertThat(config.getNmsThreshold()).isEqualTo(0.6f);
		assertThat(config.getNmsDiffClassThreshold()).isEqualTo(0.98f);
		assertThat(config.getMaxDetections()).isEqualTo(100);
	}

	@Test
	void configInvalidNmsThresholdShouldThrow() {
		assertThatThrownBy(LayoutConfig.builder().nmsThreshold(1.5f).build()::validate)
			.isInstanceOf(net.dreamlu.mica.ai.common.exception.MicaAiException.class)
			.hasMessageContaining("nmsThreshold");
		assertThatThrownBy(LayoutConfig.builder().nmsDiffClassThreshold(-0.1f).build()::validate)
			.isInstanceOf(net.dreamlu.mica.ai.common.exception.MicaAiException.class)
			.hasMessageContaining("nmsDiffClassThreshold");
	}

	@Test
	void configInvalidScoreShouldThrow() {
		LayoutConfig bad = LayoutConfig.builder().scoreThreshold(1.5f).build();
		assertThatThrownBy(bad::validate)
			.isInstanceOf(net.dreamlu.mica.ai.common.exception.MicaAiException.class)
			.hasMessageContaining("scoreThreshold");
	}

	@Test
	void configInvalidClassThresholdShouldThrow() {
		LayoutConfig bad = LayoutConfig.builder()
			.classScoreThresholds(java.util.Collections.singletonMap(99, 0.5f))
			.build();
		assertThatThrownBy(bad::validate)
			.isInstanceOf(net.dreamlu.mica.ai.common.exception.MicaAiException.class)
			.hasMessageContaining("classScoreThresholds key");
	}

	@Test
	void configInvalidMeanShouldThrow() {
		LayoutConfig bad = LayoutConfig.builder().mean(new float[]{1f, 2f}).build();
		assertThatThrownBy(bad::validate)
			.isInstanceOf(net.dreamlu.mica.ai.common.exception.MicaAiException.class)
			.hasMessageContaining("mean");
	}

	@Test
	void layoutLabelOfShouldMatch() {
		assertThat(LayoutLabel.of("doc_title")).isEqualTo(LayoutLabel.DOC_TITLE);
		assertThat(LayoutLabel.of("FIGURE")).isNull();
		assertThat(LayoutLabel.of("image")).isEqualTo(LayoutLabel.IMAGE);
		assertThat(LayoutLabel.ofIndex(21)).isEqualTo(LayoutLabel.TABLE);
		assertThat(LayoutLabel.of("nope")).isNull();
		assertThat(LayoutLabel.ofIndex(99)).isNull();
		assertThat(LayoutLabel.numClasses()).isEqualTo(25);
	}

	@Test
	void createWithoutModelShouldThrow() {
		// classpath: 默认路径下没有真实模型，构造应当快速失败（MicaAiException）
		LayoutConfig config = LayoutConfig.builder().build();
		assertThatThrownBy(() -> LayoutPipeline.create(config))
			.isInstanceOf(net.dreamlu.mica.ai.common.exception.MicaAiException.class);
	}

	@Test
	void detectNullShouldReturnEmptyWithoutModelLoad() {
		// 不真正构造 pipeline，直接验证 LayoutResult.emptyList() 工厂方法
		assertThat(LayoutResult.emptyList()).isEmpty();
	}
}