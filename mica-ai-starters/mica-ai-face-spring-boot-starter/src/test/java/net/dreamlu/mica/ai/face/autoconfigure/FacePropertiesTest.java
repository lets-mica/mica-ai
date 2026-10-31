/*
 * Copyright (c) 2019-2029, Dreamlu 卢春梦 (596392912@qq.com & dreamlu.net).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.dreamlu.mica.ai.face.autoconfigure;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import net.dreamlu.mica.ai.face.avatar.AvatarOptions;
import net.dreamlu.mica.ai.face.card.CardOptions;
import net.dreamlu.mica.ai.face.config.ExpressionConfig;
import net.dreamlu.mica.ai.face.config.FaceQualityConfig;
import net.dreamlu.mica.ai.face.config.ModelConfig;
import net.dreamlu.mica.ai.face.config.MultiFaceStrategy;
import org.junit.jupiter.api.Test;
import org.springframework.boot.context.properties.bind.Bindable;
import org.springframework.boot.context.properties.bind.Binder;
import org.springframework.boot.context.properties.source.MapConfigurationPropertySource;

import java.util.HashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link FaceProperties} 配置绑定测试（纯绑定，不启动 Spring 上下文、不加载模型）。
 *
 * <p>验证目标：文档里写的 {@code mica.ai.face.*} 键名真的能绑上、
 * 默认值与 {@link ModelConfig} 保持一致、五个模型路径与各类参数都被装配。
 */
class FacePropertiesTest {

	private static FaceProperties bind(Map<String, Object> source) {
		Binder binder = new Binder(new MapConfigurationPropertySource(source));
		return binder.bind("mica.ai.face", Bindable.of(FaceProperties.class))
			.orElseGet(FaceProperties::new);
	}

	@Test
	void defaultsShouldMatchCoreConfigDefaults() {
		FaceProperties p = new FaceProperties();
		ModelConfig c = ModelConfig.builder().build();

		assertThat(p.isEnabled()).as("默认应启用").isTrue();
		assertThat(p.getDetection().getThreshold()).isEqualTo(c.getDetectionThreshold()).isEqualTo(0.9f);
		assertThat(p.getDetection().getNmsThreshold()).isEqualTo(c.getNmsThreshold()).isEqualTo(0.3f);
		assertThat(p.getVerify().getThreshold()).isEqualTo(c.getVerifyThreshold()).isEqualTo(0.35f);
		assertThat(p.getVerify().getStrategy()).isEqualTo(c.getMultiFaceStrategy())
			.isEqualTo(MultiFaceStrategy.DEFAULT);
		assertThat(p.getLiveness().isEnabled()).as("活体默认开启").isTrue();
		assertThat(p.getLiveness().getThreshold()).isEqualTo(0.85f);
		assertThat(p.getLiveness().getCropScale()).isEqualTo(2.7);
		assertThat(p.getAvatar().getSize()).isEqualTo(256);
		assertThat(p.getCard().getOutputWidth()).isEqualTo(1011);
		assertThat(p.getQuality().isEnabled()).as("质量门控默认关闭").isFalse();
		assertThat(p.getExpression().isEnabled()).as("表情识别默认关闭").isFalse();
		assertThat(p.getOnnx().getDevice()).as("默认走 CPU").isEqualTo(OrtDevice.CPU);
	}

	@Test
	void kebabCaseKeysShouldBind() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.face.enabled", "false");
		src.put("mica.ai.face.model.detection.path", "model-tools/face/models/face_detection_yunet_2023mar.onnx");
		src.put("mica.ai.face.model.recognition.path", "model-tools/face/models/face_recognition_sface_2021dec.onnx");
		src.put("mica.ai.face.model.liveness.path", "model-tools/face/models/antispoof_model.onnx");
		src.put("mica.ai.face.model.quality.path", "model-tools/face/models/ediffiqa_tiny_jun2024.onnx");
		src.put("mica.ai.face.model.expression.path",
			"model-tools/face/models/facial_expression_recognition_mobilefacenet_2022july.onnx");
		src.put("mica.ai.face.detection.threshold", "0.8");
		src.put("mica.ai.face.detection.nms-threshold", "0.4");
		src.put("mica.ai.face.liveness.enabled", "false");
		src.put("mica.ai.face.liveness.threshold", "0.7");
		src.put("mica.ai.face.liveness.crop-scale", "3.0");
		src.put("mica.ai.face.verify.threshold", "0.5");
		src.put("mica.ai.face.verify.strategy", "REJECT");
		src.put("mica.ai.face.avatar.size", "512");
		src.put("mica.ai.face.avatar.face-scale", "1.3");
		src.put("mica.ai.face.avatar.background", "#000000");
		src.put("mica.ai.face.card.output-width", "800");
		src.put("mica.ai.face.card.enhance", "false");
		src.put("mica.ai.face.quality.enabled", "true");
		src.put("mica.ai.face.quality.threshold", "0.7");
		src.put("mica.ai.face.expression.enabled", "true");
		src.put("mica.ai.face.onnx.device", "gpu");
		src.put("mica.ai.face.onnx.intra-op-num-threads", "4");

		FaceProperties p = bind(src);

		assertThat(p.isEnabled()).isFalse();
		assertThat(p.getModel().getDetection().getPath())
			.isEqualTo("model-tools/face/models/face_detection_yunet_2023mar.onnx");
		assertThat(p.getModel().getRecognition().getPath())
			.isEqualTo("model-tools/face/models/face_recognition_sface_2021dec.onnx");
		assertThat(p.getModel().getLiveness().getPath())
			.isEqualTo("model-tools/face/models/antispoof_model.onnx");
		assertThat(p.getModel().getQuality().getPath())
			.isEqualTo("model-tools/face/models/ediffiqa_tiny_jun2024.onnx");
		assertThat(p.getModel().getExpression().getPath())
			.isEqualTo("model-tools/face/models/facial_expression_recognition_mobilefacenet_2022july.onnx");
		assertThat(p.getDetection().getThreshold()).isEqualTo(0.8f);
		assertThat(p.getDetection().getNmsThreshold()).isEqualTo(0.4f);
		assertThat(p.getLiveness().isEnabled()).isFalse();
		assertThat(p.getLiveness().getThreshold()).isEqualTo(0.7f);
		assertThat(p.getLiveness().getCropScale()).isEqualTo(3.0);
		assertThat(p.getVerify().getThreshold()).isEqualTo(0.5f);
		assertThat(p.getVerify().getStrategy()).isEqualTo(MultiFaceStrategy.REJECT);
		assertThat(p.getAvatar().getSize()).isEqualTo(512);
		assertThat(p.getAvatar().getFaceScale()).isEqualTo(1.3);
		assertThat(p.getAvatar().getBackground()).isEqualTo("#000000");
		assertThat(p.getCard().getOutputWidth()).isEqualTo(800);
		assertThat(p.getCard().isEnhance()).isFalse();
		assertThat(p.getQuality().isEnabled()).isTrue();
		assertThat(p.getQuality().getThreshold()).isEqualTo(0.7f);
		assertThat(p.getExpression().isEnabled()).isTrue();
		assertThat(p.getOnnx().getDevice()).isEqualTo(OrtDevice.GPU);
		assertThat(p.getOnnx().getIntraOpNumThreads()).isEqualTo(4);
	}

	/**
	 * 装配一致性：{@link FaceAutoConfiguration#toModelConfig} 必须把每个属性都装进
	 * {@link ModelConfig}；活体关闭时不应携带活体模型路径。
	 */
	@Test
	void autoConfigurationShouldWireModelConfig() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.face.model.detection.path", "/tmp/det.onnx");
		src.put("mica.ai.face.model.recognition.path", "/tmp/rec.onnx");
		src.put("mica.ai.face.model.liveness.path", "/tmp/live.onnx");
		src.put("mica.ai.face.detection.threshold", "0.8");
		src.put("mica.ai.face.detection.nms-threshold", "0.4");
		src.put("mica.ai.face.verify.threshold", "0.5");
		src.put("mica.ai.face.verify.strategy", "REJECT");
		src.put("mica.ai.face.onnx.device", "gpu");

		FaceProperties p = bind(src);
		ModelConfig c = FaceAutoConfiguration.toModelConfig(p);

		assertThat(c.getDetectionModelPath()).isEqualTo("/tmp/det.onnx");
		assertThat(c.getRecognitionModelPath()).isEqualTo("/tmp/rec.onnx");
		assertThat(c.getLivenessModelPath()).isEqualTo("/tmp/live.onnx");
		assertThat(c.getDetectionThreshold()).isEqualTo(0.8f);
		assertThat(c.getNmsThreshold()).isEqualTo(0.4f);
		assertThat(c.getVerifyThreshold()).isEqualTo(0.5f);
		assertThat(c.getMultiFaceStrategy()).isEqualTo(MultiFaceStrategy.REJECT);
		assertThat(c.getOnnx()).isSameAs(p.getOnnx());
	}

	/**
	 * 活体关闭时不应再要求活体模型路径。
	 */
	@Test
	void livenessDisabledShouldDropLivenessModelPath() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.face.model.liveness.path", "/tmp/live.onnx");
		src.put("mica.ai.face.liveness.enabled", "false");

		ModelConfig c = FaceAutoConfiguration.toModelConfig(bind(src));
		assertThat(c.getLivenessModelPath()).isNull();
	}

	/**
	 * 头像 / 卡片参数必须被完整装配进核心值对象。
	 */
	@Test
	void autoConfigurationShouldWireAvatarAndCardOptions() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.face.avatar.size", "512");
		src.put("mica.ai.face.avatar.face-scale", "1.3");
		src.put("mica.ai.face.avatar.vertical-offset", "0.2");
		src.put("mica.ai.face.avatar.de-rotate", "false");
		src.put("mica.ai.face.avatar.rotation-degrees", "15.0");
		src.put("mica.ai.face.avatar.auto-orient", "false");
		src.put("mica.ai.face.avatar.background", "#000000");
		src.put("mica.ai.face.avatar.tile-detect", "false");
		src.put("mica.ai.face.avatar.tile-size", "320");
		src.put("mica.ai.face.avatar.tile-overlap", "0.2");
		src.put("mica.ai.face.avatar.tile-threshold", "0.7");
		src.put("mica.ai.face.avatar.min-face-size", "60");
		src.put("mica.ai.face.avatar.max-faces", "3");
		src.put("mica.ai.face.card.output-width", "800");
		src.put("mica.ai.face.card.output-height", "500");
		src.put("mica.ai.face.card.aspect-tolerance", "0.2");
		src.put("mica.ai.face.card.enhance", "false");
		src.put("mica.ai.face.card.sharpen-amount", "1.0");
		src.put("mica.ai.face.card.sharpen-sigma", "1.0");
		src.put("mica.ai.face.card.min-card-size", "300");

		FaceProperties p = bind(src);
		AvatarOptions avatar = FaceAutoConfiguration.toAvatarOptions(p);
		CardOptions card = FaceAutoConfiguration.toCardOptions(p);

		assertThat(avatar.getSize()).isEqualTo(512);
		assertThat(avatar.getFaceScale()).isEqualTo(1.3);
		assertThat(avatar.getVerticalOffset()).isEqualTo(0.2);
		assertThat(avatar.isDeRotate()).isFalse();
		assertThat(avatar.getRotationDegrees()).isEqualTo(15.0);
		assertThat(avatar.isAutoOrient()).isFalse();
		assertThat(avatar.getBackground()).isEqualTo("#000000");
		assertThat(avatar.isTileDetect()).isFalse();
		assertThat(avatar.getTileSize()).isEqualTo(320);
		assertThat(avatar.getTileOverlap()).isEqualTo(0.2);
		assertThat(avatar.getTileThreshold()).isEqualTo(0.7);
		assertThat(avatar.getMinFaceSize()).isEqualTo(60);
		assertThat(avatar.getMaxFaces()).isEqualTo(3);

		assertThat(card.getOutputWidth()).isEqualTo(800);
		assertThat(card.getOutputHeight()).isEqualTo(500);
		assertThat(card.getAspectTolerance()).isEqualTo(0.2);
		assertThat(card.isEnhance()).isFalse();
		assertThat(card.getSharpenAmount()).isEqualTo(1.0);
		assertThat(card.getSharpenSigma()).isEqualTo(1.0);
		assertThat(card.getMinCardSize()).isEqualTo(300);
	}

	/**
	 * 质量评估 / 表情识别的模型路径必须被装配，且未配置时启动即失败。
	 */
	@Test
	void autoConfigurationShouldWireQualityAndExpressionConfig() {
		Map<String, Object> src = new HashMap<>();
		src.put("mica.ai.face.model.quality.path", "/tmp/quality.onnx");
		src.put("mica.ai.face.model.expression.path", "/tmp/expression.onnx");
		src.put("mica.ai.face.quality.threshold", "0.7");
		src.put("mica.ai.face.onnx.device", "gpu");

		FaceProperties p = bind(src);
		FaceQualityConfig quality = FaceAutoConfiguration.toQualityConfig(p);
		ExpressionConfig expression = FaceAutoConfiguration.toExpressionConfig(p);

		assertThat(quality.getModelPath()).isEqualTo("/tmp/quality.onnx");
		assertThat(quality.getThreshold()).isEqualTo(0.7f);
		assertThat(quality.getOnnx()).isSameAs(p.getOnnx());
		quality.validate();

		assertThat(expression.getModelPath()).isEqualTo("/tmp/expression.onnx");
		assertThat(expression.getOnnx()).isSameAs(p.getOnnx());
		expression.validate();
	}

	/**
	 * 模型不随 jar 分发，没有内置 classpath 兜底：未配置模型路径必须启动即失败。
	 */
	@Test
	void unsetModelPathsShouldFailFast() {
		assertThatThrownBy(() -> FaceQualityConfig.builder().build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");

		assertThatThrownBy(() -> ExpressionConfig.builder().build().validate())
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("modelPath");
	}
}
