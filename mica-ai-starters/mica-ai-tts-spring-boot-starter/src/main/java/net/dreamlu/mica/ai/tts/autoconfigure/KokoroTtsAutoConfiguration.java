package net.dreamlu.mica.ai.tts.autoconfigure;

import net.dreamlu.mica.ai.tts.engine.KokoroTts;
import net.dreamlu.mica.ai.tts.config.KokoroTtsConfig;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Kokoro TTS 自动配置。
 */
@AutoConfiguration
@ConditionalOnClass(KokoroTts.class)
@EnableConfigurationProperties(KokoroTtsProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.tts", name = "model-path")
public class KokoroTtsAutoConfiguration {

	@Bean
	public KokoroTtsConfig kokoroTtsConfig(KokoroTtsProperties properties) {
		return KokoroTtsConfig.builder()
			.modelPath(properties.getModelPath())
			.voicesDir(properties.getVoicesDir())
			.configPath(properties.getConfigPath())
			.defaultVoice(properties.getDefaultVoice())
			.defaultSpeed(properties.getDefaultSpeed())
			.onnxProvider(properties.getOnnxProvider())
			.build();
	}

	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public KokoroTts kokoroTts(KokoroTtsConfig kokoroTtsConfig) throws Exception {
		return new KokoroTts(kokoroTtsConfig);
	}
}
