package net.dreamlu.mica.ai.voice.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
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
 *
 * <p>启用条件：{@code mica.ai.voice.enabled=true}（默认）。
 * 启用后必填项（{@code encoder-path} / {@code decoder-path} / {@code tokenizer-path}）缺失将启动失败。
 */
@AutoConfiguration
@ConditionalOnClass(SenseVoice.class)
@EnableConfigurationProperties(SenseVoiceProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.voice", name = "enabled", havingValue = "true", matchIfMissing = true)
public class SenseVoiceAutoConfiguration {

	@Bean
	public SenseVoiceConfig senseVoiceConfig(SenseVoiceProperties properties) {
		requireNonBlank(properties.getEncoderPath(), "mica.ai.voice.encoder-path");
		requireNonBlank(properties.getDecoderPath(), "mica.ai.voice.decoder-path");
		requireNonBlank(properties.getTokenizerPath(), "mica.ai.voice.tokenizer-path");
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

	@Bean
	@ConditionalOnMissingBean
	public SenseVoice senseVoice(SenseVoiceConfig senseVoiceConfig) {
		return new SenseVoice(senseVoiceConfig);
	}

	private static void requireNonBlank(String value, String name) {
		if (value == null || value.isBlank()) {
			throw new MicaAiException(
				"mica-ai-voice 启用失败：[" + name + "] 必须配置（可在 application.yml 中设置 mica.ai.voice.enabled=false 关闭该 Starter）");
		}
	}
}
