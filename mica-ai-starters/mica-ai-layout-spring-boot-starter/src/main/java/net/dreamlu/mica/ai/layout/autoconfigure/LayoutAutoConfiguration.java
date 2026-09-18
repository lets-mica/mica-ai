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

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.layout.config.LayoutConfig;
import net.dreamlu.mica.ai.layout.model.LayoutLabel;
import net.dreamlu.mica.ai.layout.pipeline.LayoutPipeline;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * mica-ai-layout Spring Boot 自动装配（基于 mica-auto）。
 */
@Slf4j
@Configuration
@EnableConfigurationProperties(LayoutProperties.class)
@ConditionalOnClass(LayoutPipeline.class)
@ConditionalOnProperty(prefix = "mica.ai.layout", name = "enabled", havingValue = "true", matchIfMissing = true)
public class LayoutAutoConfiguration {

	/**
	 * 创建文档版面分析管道 Bean。
	 *
	 * @param properties 版面分析配置属性
	 * @return LayoutPipeline 实例
	 */
	@Bean
	@ConditionalOnMissingBean
	public LayoutPipeline layoutPipeline(LayoutProperties properties) {
		LayoutConfig config = LayoutConfig.builder()
			.modelVersion(properties.getModelVersion())
			.modelPath(properties.getModelPath())
			.maxSideLength(properties.getMaxSideLength())
			.scoreThreshold(properties.getScoreThreshold())
			.scoreRatio(properties.getScoreRatio())
			.classScoreThresholds(properties.getClassScoreThresholds())
			.layoutNms(properties.isLayoutNms())
			.nmsThreshold(properties.getNmsThreshold())
			.nmsDiffClassThreshold(properties.getNmsDiffClassThreshold())
			.maxDetections(properties.getMaxDetections())
			.skipOrderLabels(resolveSkipOrderLabels(properties.getSkipOrderLabels()))
			.mean(properties.getMean())
			.std(properties.getStd())
			.onnx(properties.getOnnx())
			.build();
		return LayoutPipeline.create(config);
	}

	/**
	 * 把配置里的标签 code 列表解析为 {@link LayoutLabel} 集合。
	 *
	 * <p>未配置（{@code null}）时使用 PaddleX 默认名单；显式配置空列表表示
	 * 「所有类别都参与阅读顺序编号」。遇到未识别的 code 直接 fail-fast，
	 * 避免拼错的标签名静默失效。
	 *
	 * @param codes 配置的标签 code 列表；{@code null} 表示走默认名单
	 * @return 解析后的标签集合
	 */
	static Set<LayoutLabel> resolveSkipOrderLabels(List<String> codes) {
		if (codes == null) {
			return LayoutLabel.defaultSkipOrderLabels();
		}
		Set<LayoutLabel> set = new LinkedHashSet<>(codes.size());
		for (String code : codes) {
			LayoutLabel label = LayoutLabel.of(code);
			if (label == null) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"mica.ai.layout.skip-order-labels 含未识别的标签 code: " + code);
			}
			set.add(label);
		}
		return set;
	}
}