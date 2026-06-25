/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.autoconfigure;

import net.dreamlu.mica.ai.face.FaceEngine;
import net.dreamlu.mica.ai.face.config.FaceConfig;
import net.dreamlu.mica.ai.face.engine.FaceDetector;
import net.dreamlu.mica.ai.face.engine.FaceRecognizer;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Face 引擎自动配置。
 *
 * <p>本 Starter 暴露 {@link FaceEngine} Bean，模型实现走 OpenCV Zoo YuNet + SFace（Apache-2.0）。
 * 人脸库入库与 1:N 检索由调用方自行使用向量数据库（Milvus / pgvector / Qdrant 等）实现。
 *
 * <p>如果用户已经声明了 {@link FaceDetector} 或 {@link FaceRecognizer} 的 Bean，Starter 会自动注入而非创建默认实现。
 *
 * @since 1.0.0
 */
@AutoConfiguration
@ConditionalOnClass(FaceEngine.class)
@EnableConfigurationProperties(FaceProperties.class)
@ConditionalOnProperty(prefix = "mica.ai.face", name = {"det-model-path", "rec-model-path"})
public class FaceAutoConfiguration {

	@Bean(destroyMethod = "close")
	@ConditionalOnMissingBean
	public FaceEngine faceEngine(FaceProperties properties,
								 ObjectProvider<FaceDetector> detectorProvider,
								 ObjectProvider<FaceRecognizer> recognizerProvider) {
		FaceConfig config = FaceConfig.builder()
			.modelType(toModelType(properties.getModelType()))
			.detModelPath(properties.getDetModelPath())
			.recModelPath(properties.getRecModelPath())
			.detScoreThreshold(properties.getDetScoreThreshold())
			.detNmsThreshold(properties.getDetNmsThreshold())
			.intraOpNumThreads(properties.getIntraOpNumThreads())
			.interOpNumThreads(properties.getInterOpNumThreads())
			.build();

		FaceDetector detector = detectorProvider.getIfAvailable(() -> null);
		FaceRecognizer recognizer = recognizerProvider.getIfAvailable(() -> null);

		return FaceEngine.builder()
			.config(config)
			.detector(detector)
			.recognizer(recognizer)
			.build();
	}

	private static FaceConfig.ModelType toModelType(FaceProperties.FaceConfigModelType type) {
		if (type == null) {
			return FaceConfig.ModelType.YUNET_SFACE;
		}
		return switch (type) {
			case YUNET_SFACE -> FaceConfig.ModelType.YUNET_SFACE;
		};
	}
}