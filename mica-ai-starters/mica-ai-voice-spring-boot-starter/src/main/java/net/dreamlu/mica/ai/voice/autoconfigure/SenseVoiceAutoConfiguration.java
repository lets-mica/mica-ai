package net.dreamlu.mica.ai.voice.autoconfigure;

import net.dreamlu.mica.ai.voice.engine.SenseVoice;
import net.dreamlu.mica.ai.voice.config.SenseVoiceConfig;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * SenseVoice 自动配置。
 */
@AutoConfiguration
@ConditionalOnClass(SenseVoice.class)
@EnableConfigurationProperties(SenseVoiceProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.voice", name = "encoder-path")
public class SenseVoiceAutoConfiguration {

	@Bean
	public SenseVoiceConfig senseVoiceConfig(SenseVoiceProperties properties) {
		return SenseVoiceConfig.builder()
			.encoderPath(properties.getEncoderPath())
			.decoderPath(properties.getDecoderPath())
			.tokenizerPath(properties.getTokenizerPath())
			.itn(properties.isItn())
			.topK(properties.getTopK())
			.hotwords(properties.getHotwords())
			.onnxProvider(properties.getOnnxProvider())
			.build();
	}

	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public SenseVoice senseVoice(SenseVoiceConfig senseVoiceConfig) {
		return new SenseVoice(senseVoiceConfig);
	}
}
