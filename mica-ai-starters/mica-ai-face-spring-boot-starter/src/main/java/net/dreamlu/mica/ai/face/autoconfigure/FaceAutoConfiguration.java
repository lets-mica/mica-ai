/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.autoconfigure;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.face.alignment.FaceAligner;
import net.dreamlu.mica.ai.face.avatar.AvatarExtractor;
import net.dreamlu.mica.ai.face.avatar.AvatarOptions;
import net.dreamlu.mica.ai.face.card.CardExtractor;
import net.dreamlu.mica.ai.face.card.CardOptions;
import net.dreamlu.mica.ai.face.config.ModelConfig;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.liveness.LivenessDetector;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.face.recognition.FeatureExtractor;
import net.dreamlu.mica.ai.face.verification.FaceVerifier;
import nu.pattern.OpenCV;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.stereotype.Component;

/**
 * mica-ai-face Spring Boot 自动装配（基于 mica-auto）。
 *
 * <p>由 {@code mica-auto} 扫描本类上的 {@link Component} 注解，自动生成
 * {@code META-INF/spring.factories} 中的 {@code EnableAutoConfiguration} 条目。
 */
@Slf4j
@Component
@EnableConfigurationProperties(FaceProperties.class)
@ConditionalOnClass(FaceDetector.class)
public class FaceAutoConfiguration implements InitializingBean {

	@Bean(destroyMethod = "destroy")
	@ConditionalOnMissingBean
	public ModelManager modelManager(FaceProperties properties) {
		FaceProperties.Model model = properties.getModel();
		ModelConfig config = ModelConfig.builder()
			.detectionModelPath(model.getDetection().getPath())
			.recognitionModelPath(model.getRecognition().getPath())
			.livenessModelPath(properties.getLiveness().isEnabled() ? model.getLiveness().getPath() : null)
			.detectionThreshold(properties.getDetection().getThreshold())
			.nmsThreshold(properties.getDetection().getNmsThreshold())
			.verifyThreshold(properties.getVerify().getThreshold())
			.onnx(properties.getOnnx())
			.build();
		return ModelManager.create(config);
	}

	@Bean
	@ConditionalOnMissingBean
	public FaceDetector faceDetector(ModelManager modelManager) {
		return new FaceDetector(modelManager);
	}

	@Bean
	@ConditionalOnMissingBean
	public FaceAligner faceAligner() {
		return new FaceAligner();
	}

	@Bean
	@ConditionalOnMissingBean
	public FeatureExtractor featureExtractor(ModelManager modelManager) {
		return new FeatureExtractor(modelManager);
	}

	@Bean
	@ConditionalOnMissingBean
	@ConditionalOnProperty(prefix = "mica.ai.face.liveness", name = "enabled", havingValue = "true", matchIfMissing = true)
	public LivenessDetector livenessDetector(ModelManager modelManager, FaceProperties properties) {
		return new LivenessDetector(modelManager,
			properties.getLiveness().getThreshold(),
			properties.getLiveness().getCropScale());
	}

	@Bean
	@ConditionalOnMissingBean
	public FaceVerifier faceVerifier(ModelManager modelManager) {
		return new FaceVerifier(modelManager);
	}

	@Bean
	@ConditionalOnMissingBean
	public AvatarExtractor avatarExtractor(FaceDetector faceDetector, FaceProperties properties) {
		FaceProperties.Avatar a = properties.getAvatar();
		AvatarOptions defaults = AvatarOptions.builder()
			.size(a.getSize())
			.faceScale(a.getFaceScale())
			.verticalOffset(a.getVerticalOffset())
			.deRotate(a.isDeRotate())
			.rotationDegrees(a.getRotationDegrees())
			.autoOrient(a.isAutoOrient())
			.background(a.getBackground())
			.tileDetect(a.isTileDetect())
			.tileSize(a.getTileSize())
			.tileOverlap(a.getTileOverlap())
			.tileThreshold(a.getTileThreshold())
			.minFaceSize(a.getMinFaceSize())
			.maxFaces(a.getMaxFaces())
			.build();
		return new AvatarExtractor(faceDetector, defaults);
	}

	@Bean
	@ConditionalOnMissingBean
	public CardExtractor cardExtractor(FaceProperties properties) {
		FaceProperties.Card c = properties.getCard();
		CardOptions.CardOptionsBuilder builder = CardOptions.defaults().toBuilder()
			.outputWidth(c.getOutputWidth())
			.outputHeight(c.getOutputHeight())
			.aspectTolerance(c.getAspectTolerance())
			.enhance(c.isEnhance())
			.sharpenAmount(c.getSharpenAmount())
			.sharpenSigma(c.getSharpenSigma())
			.minCardSize(c.getMinCardSize());
		// 默认不注入 FaceDetector；需要 autoOrient 时由业务模块自行提供带检测器的 CardExtractor
		return new CardExtractor(builder.build());
	}

	@Override
	public void afterPropertiesSet() {
		try {
			OpenCV.loadLocally();
			log.info("mica-ai-face: OpenCV 原生库加载完成");
		} catch (Throwable t) {
			log.warn("mica-ai-face: OpenCV 原生库加载失败 - {}", t.getMessage());
		}
	}
}
