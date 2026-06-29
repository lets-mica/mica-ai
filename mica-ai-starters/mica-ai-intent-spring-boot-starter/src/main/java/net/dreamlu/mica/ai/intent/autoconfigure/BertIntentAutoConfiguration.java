package net.dreamlu.mica.ai.intent.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.intent.engine.BertIntent;
import net.dreamlu.mica.ai.intent.config.BertIntentConfig;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

import java.io.IOException;
import java.util.List;

/**
 * BERT 意图识别自动配置。
 *
 * <p>启用条件：{@code mica.ai.intent.enabled=true}（默认）。
 * 启用后必填项（{@code model-path} / {@code vocab-path} / {@code labels}）缺失将启动失败。
 */
@AutoConfiguration
@ConditionalOnClass(BertIntent.class)
@EnableConfigurationProperties(BertIntentProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.intent", name = "enabled", havingValue = "true", matchIfMissing = true)
public class BertIntentAutoConfiguration {

	@Bean
	public BertIntentConfig bertIntentConfig(BertIntentProperties properties) {
		requireNonBlank(properties.getModelPath(), "mica.ai.intent.model-path");
		requireNonBlank(properties.getVocabPath(), "mica.ai.intent.vocab-path");
		List<String> labels = properties.getLabels();
		if (labels == null || labels.isEmpty()) {
			throw new MicaAiException(
				"mica-ai-intent 启用失败：[mica.ai.intent.labels] 必须配置且非空");
		}
		return BertIntentConfig.builder()
			.modelPath(properties.getModelPath())
			.vocabPath(properties.getVocabPath())
			.maxLength(properties.getMaxLength())
			.labels(labels)
			.intraOpNumThreads(properties.getIntraOpNumThreads())
			.interOpNumThreads(properties.getInterOpNumThreads())
			.build();
	}

	@Bean
	@ConditionalOnMissingBean
	public BertIntent bertIntent(BertIntentConfig config) throws IOException {
		return new BertIntent(config);
	}

	private static void requireNonBlank(String value, String name) {
		if (value == null || value.isBlank()) {
			throw new MicaAiException(
				"mica-ai-intent 启用失败：[" + name + "] 必须配置（可在 application.yml 中设置 mica.ai.intent.enabled=false 关闭该 Starter）");
		}
	}
}
