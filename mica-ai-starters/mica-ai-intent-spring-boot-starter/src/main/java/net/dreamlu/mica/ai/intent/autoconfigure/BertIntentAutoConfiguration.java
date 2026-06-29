/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent.autoconfigure;

import net.dreamlu.mica.ai.intent.engine.BertIntent;
import net.dreamlu.mica.ai.intent.config.BertIntentConfig;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

import java.io.IOException;

/**
 * BERT 意图识别自动配置。
 */
@AutoConfiguration
@ConditionalOnClass(BertIntent.class)
@EnableConfigurationProperties(BertIntentProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.intent", name = "model-path")
public class BertIntentAutoConfiguration {

	@Bean
	public BertIntentConfig bertIntentConfig(BertIntentProperties properties) {
		return BertIntentConfig.builder()
			.modelPath(properties.getModelPath())
			.vocabPath(properties.getVocabPath())
			.maxLength(properties.getMaxLength())
			.labels(properties.getLabels())
			.intraOpNumThreads(properties.getIntraOpNumThreads())
			.interOpNumThreads(properties.getInterOpNumThreads())
			.build();
	}

	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public BertIntent bertIntent(BertIntentConfig config) throws IOException {
		return new BertIntent(config);
	}
}
