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
package net.dreamlu.mica.ai.layout.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import net.dreamlu.mica.ai.layout.config.LayoutConfig;
import net.dreamlu.mica.ai.layout.model.LayoutLabel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link LayoutProperties} 配置绑定测试（纯绑定，不启动 Spring 上下文、不加载模型）。
 *
 * <p>验证目标：文档里写的 {@code mica.ai.layout.*} 键名真的能绑上、
 * 默认值与 {@link LayoutConfig} 保持一致、阅读顺序名单能正确解析成标签枚举。
 */
class LayoutPropertiesTest {

	private static LayoutProperties bind(Map<String, Object> source) {
		Binder binder = new Binder(new MapConfigurationPropertySource(source));
		return binder.bind("mica.ai.layout", Bindable.of(LayoutProperties.class))
			.orElseGet(LayoutProperties::new);
	}

	@Test
	void defaultsShouldMatchCoreConfigDefaults() {
		LayoutProperties p = new LayoutProperties();
		LayoutConfig c = LayoutConfig.builder().build();

		assertThat(p.isEnabled()).as("默认应启用").isTrue();
		assertThat(p.getModelVersion()).isEqualTo(c.getModelVersion()).isEqualTo("v3");
		assertThat(p.getMaxSideLength()).isEqualTo(c.getMaxSideLength()).isEqualTo(800);
		assertThat(p.getScoreThreshold()).isEqualTo(c.getScoreThreshold()).isEqualTo(0.4f);
		assertThat(p.getScoreRatio()).isEqualTo(c.getScoreRatio()).isEqualTo(0f);
		assertThat(p.isLayoutNms()).isEqualTo(c.isLayoutNms()).isTrue();
		assertThat(p.getNmsThreshold()).isEqualTo(c.getNmsThreshold()).isEqualTo(0.6f);
		assertThat(p.getNmsDiffClassThreshold()).isEqualTo(c.getNmsDiffClassThreshold())
			.isEqualTo(0.98f);
		assertThat(p.getMaxDetections()).isEqualTo(c.getMaxDetections()).isEqualTo(100);
		assertThat(p.getMean()).isEqualTo(c.getMean());
		assertThat(p.getStd()).isEqualTo(c.getStd());
		assertThat(p.getOnnx().getDevice()).as("默认走 CPU").isEqualTo(OrtDevice.CPU);
	}

	@Test
	void kebabCaseKeysShouldBind() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.layout.enabled", "false");
		src.put("mica.ai.layout.model-version", "v2");
		src.put("mica.ai.layout.model-path", "model-tools/layout/models/model.onnx");
		src.put("mica.ai.layout.max-side-length", "1024");
		src.put("mica.ai.layout.score-threshold", "0.55");
		src.put("mica.ai.layout.score-ratio", "0.6");
		src.put("mica.ai.layout.layout-nms", "false");
		src.put("mica.ai.layout.nms-threshold", "0.5");
		src.put("mica.ai.layout.nms-diff-class-threshold", "0.9");
		src.put("mica.ai.layout.max-detections", "50");
		src.put("mica.ai.layout.class-score-thresholds[0]", "0.7");
		src.put("mica.ai.layout.skip-order-labels[0]", "doc_title");
		src.put("mica.ai.layout.mean[0]", "0.5");
		src.put("mica.ai.layout.mean[1]", "0.4");
		src.put("mica.ai.layout.mean[2]", "0.3");
		src.put("mica.ai.layout.std[0]", "0.25");
		src.put("mica.ai.layout.std[1]", "0.26");
		src.put("mica.ai.layout.std[2]", "0.27");
		src.put("mica.ai.layout.onnx.device", "gpu");
		src.put("mica.ai.layout.onnx.intra-op-num-threads", "4");

		LayoutProperties p = bind(src);

		assertThat(p.isEnabled()).isFalse();
		assertThat(p.getModelVersion()).isEqualTo("v2");
		assertThat(p.getModelPath()).isEqualTo("model-tools/layout/models/model.onnx");
		assertThat(p.getMaxSideLength()).isEqualTo(1024);
		assertThat(p.getScoreThreshold()).isEqualTo(0.55f);
		assertThat(p.getScoreRatio()).isEqualTo(0.6f);
		assertThat(p.isLayoutNms()).isFalse();
		assertThat(p.getNmsThreshold()).isEqualTo(0.5f);
		assertThat(p.getNmsDiffClassThreshold()).isEqualTo(0.9f);
		assertThat(p.getMaxDetections()).isEqualTo(50);
		assertThat(p.getClassScoreThresholds()).containsEntry(0, 0.7f);
		assertThat(p.getSkipOrderLabels()).containsExactly("doc_title");
		assertThat(p.getMean()).containsExactly(0.5f, 0.4f, 0.3f);
		assertThat(p.getStd()).containsExactly(0.25f, 0.26f, 0.27f);
		assertThat(p.getOnnx().getDevice()).isEqualTo(OrtDevice.GPU);
		assertThat(p.getOnnx().getIntraOpNumThreads()).isEqualTo(4);
	}

	/**
	 * 装配一致性：{@link LayoutAutoConfiguration#toConfig} 必须把每个属性都装进
	 * {@link LayoutConfig}。
	 */
	@Test
	void autoConfigurationBuilderShouldWireEveryProperty() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.layout.model-version", "v2");
		src.put("mica.ai.layout.model-path", "/tmp/model.onnx");
		src.put("mica.ai.layout.max-side-length", "1024");
		src.put("mica.ai.layout.score-threshold", "0.55");
		src.put("mica.ai.layout.score-ratio", "0.6");
		src.put("mica.ai.layout.layout-nms", "false");
		src.put("mica.ai.layout.nms-threshold", "0.5");
		src.put("mica.ai.layout.nms-diff-class-threshold", "0.9");
		src.put("mica.ai.layout.max-detections", "50");
		src.put("mica.ai.layout.class-score-thresholds[0]", "0.7");
		src.put("mica.ai.layout.skip-order-labels[0]", "doc_title");
		src.put("mica.ai.layout.mean[0]", "0.5");
		src.put("mica.ai.layout.mean[1]", "0.4");
		src.put("mica.ai.layout.mean[2]", "0.3");
		src.put("mica.ai.layout.std[0]", "0.25");
		src.put("mica.ai.layout.std[1]", "0.26");
		src.put("mica.ai.layout.std[2]", "0.27");
		src.put("mica.ai.layout.onnx.device", "gpu");

		LayoutProperties p = bind(src);
		LayoutConfig c = LayoutAutoConfiguration.toConfig(p);

		assertThat(c.getModelVersion()).isEqualTo("v2");
		assertThat(c.getModelPath()).isEqualTo("/tmp/model.onnx");
		assertThat(c.getMaxSideLength()).isEqualTo(1024);
		assertThat(c.getScoreThreshold()).isEqualTo(0.55f);
		assertThat(c.getScoreRatio()).isEqualTo(0.6f);
		assertThat(c.isLayoutNms()).isFalse();
		assertThat(c.getNmsThreshold()).isEqualTo(0.5f);
		assertThat(c.getNmsDiffClassThreshold()).isEqualTo(0.9f);
		assertThat(c.getMaxDetections()).isEqualTo(50);
		assertThat(c.getClassScoreThresholds()).containsEntry(0, 0.7f);
		assertThat(c.getSkipOrderLabels()).containsExactly(LayoutLabel.DOC_TITLE);
		assertThat(c.getMean()).containsExactly(0.5f, 0.4f, 0.3f);
		assertThat(c.getStd()).containsExactly(0.25f, 0.26f, 0.27f);
		assertThat(c.getOnnx()).isSameAs(p.getOnnx());
		c.validate();
	}

	/**
	 * 数组是「整体替换」而非「按下标合并」：只写 {@code mean[0]} 会把 3 通道数组
	 * 变成 1 元素数组。这里固这个语义——用户要改归一化参数必须写全 3 个通道，
	 * 否则会在 {@code validate()} 里以「mean / std 必须为长度 3 的数组」快速失败。
	 */
	@Test
	void partialArrayBindingReplacesWholeArray() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.layout.model-path", "/tmp/model.onnx");
		src.put("mica.ai.layout.mean[0]", "0.5");

		LayoutProperties p = bind(src);
		assertThat(p.getMean()).containsExactly(0.5f);

		LayoutConfig c = LayoutAutoConfiguration.toConfig(p);
		assertThatThrownBy(c::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("mean / std");
	}

	/**
	 * 未配置阅读顺序名单时走 PaddleX 默认名单；显式配置空列表表示所有类别都参与编号。
	 */
	@Test
	void skipOrderLabelsShouldFallbackToDefault() {
		LayoutConfig defaultConfig = LayoutAutoConfiguration.toConfig(new LayoutProperties());
		assertThat(defaultConfig.getSkipOrderLabels())
			.isEqualTo(LayoutLabel.defaultSkipOrderLabels())
			.hasSize(11);

		LayoutProperties empty = new LayoutProperties();
		empty.setSkipOrderLabels(new ArrayList<>());
		LayoutConfig emptyConfig = LayoutAutoConfiguration.toConfig(empty);
		assertThat(emptyConfig.getSkipOrderLabels()).isEmpty();
	}

	/**
	 * 标签 code 拼错必须 fail-fast，避免静默失效。
	 */
	@Test
	void unknownSkipOrderLabelShouldFailFast() {
		LayoutProperties p = new LayoutProperties();
		p.setSkipOrderLabels(Collections.singletonList("doc_title_typo"));

		assertThatThrownBy(() -> LayoutAutoConfiguration.toConfig(p))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("doc_title_typo");
	}

	/**
	 * 模型 125MB，不随仓库分发：未配置模型路径必须启动即失败。
	 */
	@Test
	void unsetModelPathShouldFailFast() {
		LayoutConfig config = LayoutConfig.builder().build();
		assertThatThrownBy(config::validate)
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}
}
