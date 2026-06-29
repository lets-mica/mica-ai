package net.dreamlu.mica.ai.speaker.autoconfigure;

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
 */
@AutoConfiguration
@ConditionalOnClass(SpeakerVerifier.class)
@EnableConfigurationProperties(SpeakerVerifierProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.speaker", name = "model-path")
public class SpeakerVerifierAutoConfiguration {

	@Bean
	public SpeakerConfig speakerConfig(SpeakerVerifierProperties properties) {
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
}
