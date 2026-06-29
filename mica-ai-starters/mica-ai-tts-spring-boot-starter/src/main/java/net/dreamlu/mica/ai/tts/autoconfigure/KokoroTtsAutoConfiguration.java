package net.dreamlu.mica.ai.tts.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
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
 *
 * <p>启用条件：{@code mica.ai.tts.enabled=true}（默认）。
 * 启用后必填项（{@code model-path} / {@code voices-dir} / {@code config-path}）缺失将启动失败。
 * 若不想加载该能力，单独设置 {@code mica.ai.tts.enabled=false} 即可。
 */
@AutoConfiguration
@ConditionalOnClass(KokoroTts.class)
@EnableConfigurationProperties(KokoroTtsProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.tts", name = "enabled", havingValue = "true", matchIfMissing = true)
public class KokoroTtsAutoConfiguration {

	@Bean
	public KokoroTtsConfig kokoroTtsConfig(KokoroTtsProperties properties) {
		requireNonBlank(properties.getModelPath(), "mica.ai.tts.model-path");
		requireNonBlank(properties.getVoicesDir(), "mica.ai.tts.voices-dir");
		requireNonBlank(properties.getConfigPath(), "mica.ai.tts.config-path");
		return KokoroTtsConfig.builder()
			.modelPath(properties.getModelPath())
			.voicesDir(properties.getVoicesDir())
			.configPath(properties.getConfigPath())
			.defaultVoice(properties.getDefaultVoice())
			.defaultSpeed(properties.getDefaultSpeed())
			.onnxProvider(properties.getOnnxProvider())
			.build();
	}

	@Bean
	@ConditionalOnMissingBean
	public KokoroTts kokoroTts(KokoroTtsConfig kokoroTtsConfig) throws Exception {
		return new KokoroTts(kokoroTtsConfig);
	}

	private static void requireNonBlank(String value, String name) {
		if (value == null || value.isBlank()) {
			throw new MicaAiException(
				"mica-ai-tts 启用失败：[" + name + "] 必须配置（可在 application.yml 中设置 mica.ai.tts.enabled=false 关闭该 Starter）");
		}
	}
}
