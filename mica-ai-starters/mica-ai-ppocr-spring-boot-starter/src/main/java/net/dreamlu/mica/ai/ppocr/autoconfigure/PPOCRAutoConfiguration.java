package net.dreamlu.mica.ai.ppocr.autoconfigure;

import net.dreamlu.mica.ai.ppocr.autoconfigure.PPOCRProperties;
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
 */
@AutoConfiguration
@ConditionalOnClass(PPOcrV6Engine.class)
@EnableConfigurationProperties(PPOCRProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.ppocr", name = "det-model-path")
public class PPOCRAutoConfiguration {

	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public PPOcrV6Engine ppocrV6Engine(PPOCRProperties properties) {
		PPOcrV6Config config = PPOcrV6Config.builder()
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

		return new PPOcrV6Engine(config);
	}
}
