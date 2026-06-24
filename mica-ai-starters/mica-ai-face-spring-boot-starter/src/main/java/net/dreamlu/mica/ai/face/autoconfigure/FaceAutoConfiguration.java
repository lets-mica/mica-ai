/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.autoconfigure;

import net.dreamlu.mica.ai.face.FaceEngine;
import net.dreamlu.mica.ai.face.config.FaceConfig;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Face 引擎自动配置。
 *
 * <p>本 Starter 只暴露 {@link FaceEngine} 一个 Bean，用于把图片转成 512 维 Embedding。
 * 人脸库入库与 1:N 检索由调用方自行使用向量数据库（Milvus / pgvector / Qdrant 等）实现。
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
	public FaceEngine faceEngine(FaceProperties properties) {
		FaceConfig config = FaceConfig.builder()
			.detModelPath(properties.getDetModelPath())
			.recModelPath(properties.getRecModelPath())
			.detScoreThreshold(properties.getDetScoreThreshold())
			.detNmsThreshold(properties.getDetNmsThreshold())
			.intraOpNumThreads(properties.getIntraOpNumThreads())
			.interOpNumThreads(properties.getInterOpNumThreads())
			.build();
		return FaceEngine.create(config);
	}
}
