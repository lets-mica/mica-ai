package net.dreamlu.mica.ai.speaker.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.speaker.config.SpeakerConfig;
import net.dreamlu.mica.ai.speaker.engine.SpeakerVerifier;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Speaker Verifier 自动配置。
 *
 * <p>启用条件：{@code mica.ai.speaker.enabled=true}（默认）。
 * 启用后必填项 {@code model-path} 缺失将启动失败。
 */
@AutoConfiguration
@ConditionalOnClass(SpeakerVerifier.class)
@EnableConfigurationProperties(SpeakerVerifierProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.speaker", name = "enabled", havingValue = "true", matchIfMissing = true)
public class SpeakerVerifierAutoConfiguration {

	@Bean
	public SpeakerConfig speakerConfig(SpeakerVerifierProperties properties) {
		requireNonBlank(properties.getModelPath(), "mica.ai.speaker.model-path");
		return SpeakerConfig.builder()
			.modelPath(properties.getModelPath())
			.threshold(properties.getDefaultThreshold())
			.intraOpNumThreads(properties.getIntraOpNumThreads())
			.interOpNumThreads(properties.getInterOpNumThreads())
			.build();
	}

	@Bean
	@ConditionalOnMissingBean
	public SpeakerVerifier speakerVerifier(SpeakerConfig speakerConfig) {
		return new SpeakerVerifier(speakerConfig);
	}

	private static void requireNonBlank(String value, String name) {
		if (value == null || value.isBlank()) {
			throw new MicaAiException(
				"mica-ai-speaker 启用失败：[" + name + "] 必须配置（可在 application.yml 中设置 mica.ai.speaker.enabled=false 关闭该 Starter）");
		}
	}
}
