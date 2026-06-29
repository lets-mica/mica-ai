package net.dreamlu.mica.ai.ppocr.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.ppocr.config.PPOcrV6Config;
import net.dreamlu.mica.ai.ppocr.engine.PPOcrV6Engine;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * PP-OCR 自动配置。
 *
 * <p>启用条件：{@code mica.ai.ppocr.enabled=true}（默认）。
 * 启用后必填项（{@code det-model-path} / {@code rec-model-path} / {@code rec-char-dict-path}）缺失将启动失败。
 */
@AutoConfiguration
@ConditionalOnClass(PPOcrV6Engine.class)
@EnableConfigurationProperties(PPOCRProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.ppocr", name = "enabled", havingValue = "true", matchIfMissing = true)
public class PPOCRAutoConfiguration {

	@Bean
	public PPOcrV6Config ppocrV6Config(PPOCRProperties properties) {
		requireNonBlank(properties.getDetModelPath(), "mica.ai.ppocr.det-model-path");
		requireNonBlank(properties.getRecModelPath(), "mica.ai.ppocr.rec-model-path");
		requireNonBlank(properties.getRecCharDictPath(), "mica.ai.ppocr.rec-char-dict-path");
		return PPOcrV6Config.builder()
			.detModelPath(properties.getDetModelPath())
			.recModelPath(properties.getRecModelPath())
			.recCharDictPath(properties.getRecCharDictPath())
			.detLimitSideLen(properties.getDetLimitSideLen())
			.detLimitType(properties.getDetLimitType())
			.detMaxSideLimit(properties.getDetMaxSideLimit())
			.detThresh(properties.getDetThresh())
			.detBoxThresh(properties.getDetBoxThresh())
			.detUnclipRatio(properties.getDetUnclipRatio())
			.recImageShape(properties.getRecImageShape())
			.recBatchSize(properties.getRecBatchSize())
			.preferAccelerator(properties.isPreferAccelerator())
			.intraOpNumThreads(properties.getIntraOpNumThreads())
			.interOpNumThreads(properties.getInterOpNumThreads())
			.build();
	}

	@Bean
	@ConditionalOnMissingBean
	public PPOcrV6Engine ppocrV6Engine(PPOcrV6Config ppOcrV6Config) {
		return new PPOcrV6Engine(ppOcrV6Config);
	}

	private static void requireNonBlank(String value, String name) {
		if (value == null || value.isBlank()) {
			throw new MicaAiException(
				"mica-ai-ppocr 启用失败：[" + name + "] 必须配置（可在 application.yml 中设置 mica.ai.ppocr.enabled=false 关闭该 Starter）");
		}
	}
}
