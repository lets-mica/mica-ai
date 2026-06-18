package net.dreamlu.mica.ai.ppocr.autoconfigure;

import net.dreamlu.mica.ai.ppocr.PPOCRv6Onnx;
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
@ConditionalOnClass(PPOCRv6Onnx.class)
@EnableConfigurationProperties(PPOCRProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.ppocr", name = "det-model-path")
public class PPOCRAutoConfiguration {

	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public PPOCRv6Onnx ppocrV6Onnx(PPOCRProperties properties) {
		PPOCRv6Onnx.Config config = PPOCRv6Onnx.Config.defaults()
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
			.interOpNumThreads(properties.getInterOpNumThreads());

		return new PPOCRv6Onnx(
			properties.getDetModelPath(),
			properties.getRecModelPath(),
			properties.getRecCharDictPath(),
			config
		);
	}
}
