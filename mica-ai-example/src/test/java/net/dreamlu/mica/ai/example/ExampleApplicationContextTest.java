/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.autoconfigure.FaceAutoConfiguration;
import net.dreamlu.mica.ai.face.engine.FaceEngine;
import net.dreamlu.mica.ai.intent.autoconfigure.BertIntentAutoConfiguration;
import net.dreamlu.mica.ai.intent.engine.BertIntent;
import net.dreamlu.mica.ai.ppocr.autoconfigure.OpenCVNativeLoader;
import net.dreamlu.mica.ai.ppocr.autoconfigure.PPOCRAutoConfiguration;
import net.dreamlu.mica.ai.ppocr.engine.PPOcrV6Engine;
import net.dreamlu.mica.ai.speaker.autoconfigure.SpeakerVerifierAutoConfiguration;
import net.dreamlu.mica.ai.speaker.engine.SpeakerVerifier;
import net.dreamlu.mica.ai.tts.autoconfigure.KokoroTtsAutoConfiguration;
import net.dreamlu.mica.ai.tts.engine.KokoroTts;
import net.dreamlu.mica.ai.voice.autoconfigure.SenseVoiceAutoConfiguration;
import net.dreamlu.mica.ai.voice.engine.SenseVoice;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * mica-ai-example 集成测试：验证 6 个 Spring Boot Starter 的自动装配行为。
 *
 * <p>统一使用轻量级 {@link ApplicationContextRunner}（非 {@code @SpringBootTest}），
 * 避免 Azul Zulu 21 + Spring Boot 4.1 在 Surefire fork 下触发
 * {@code Console.istty()} native 异常。
 *
 * <p>覆盖场景：
 * <ul>
 *     <li>默认（所有 Starter {@code enabled=false}）：Context 加载成功，6 个 Engine Bean 全部不存在。</li>
 *     <li>fail-fast：{@code enabled=true} 但缺必填项时，启动抛 {@link MicaAiException}。</li>
 * </ul>
 */
class ExampleApplicationContextTest {

	@org.springframework.boot.SpringBootConfiguration
	static class EmptyConfig {
	}

	private static ApplicationContextRunner runner(Class<?>... autoConfigs) {
		return new ApplicationContextRunner()
			.withUserConfiguration(EmptyConfig.class)
			.withConfiguration(AutoConfigurations.of(autoConfigs));
	}

	@Test
	@DisplayName("默认配置：所有 Starter 关闭，Context 加载成功且无 Engine Bean")
	void disabledAllStartClean() {
		runner(
			KokoroTtsAutoConfiguration.class,
			SenseVoiceAutoConfiguration.class,
			PPOCRAutoConfiguration.class,
			SpeakerVerifierAutoConfiguration.class,
			BertIntentAutoConfiguration.class,
			FaceAutoConfiguration.class
		).withPropertyValues(
			"mica.ai.tts.enabled=false",
			"mica.ai.voice.enabled=false",
			"mica.ai.ppocr.enabled=false",
			"mica.ai.speaker.enabled=false",
			"mica.ai.intent.enabled=false",
			"mica.ai.face.enabled=false"
		).run(ctx -> {
			assertThat(ctx).hasNotFailed();
			assertNoEngine(ctx, KokoroTts.class);
			assertNoEngine(ctx, SenseVoice.class);
			assertNoEngine(ctx, PPOcrV6Engine.class);
			assertNoEngine(ctx, SpeakerVerifier.class);
			assertNoEngine(ctx, BertIntent.class);
			assertNoEngine(ctx, FaceEngine.class);
		});
	}

	@Test
	@DisplayName("TTS enabled=true 但缺 model-path：fail-fast 抛 MicaAiException")
	void ttsEnabledButMissingRequiredShouldFailFast() {
		runner(KokoroTtsAutoConfiguration.class)
			.withPropertyValues("mica.ai.tts.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class)
					.hasMessageContaining("mica.ai.tts.model-path");
			});
	}

	@Test
	@DisplayName("Voice enabled=true 但缺 encoder-path：fail-fast 抛 MicaAiException")
	void voiceEnabledButMissingRequiredShouldFailFast() {
		runner(SenseVoiceAutoConfiguration.class)
			.withPropertyValues("mica.ai.voice.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class)
					.hasMessageContaining("mica.ai.voice.encoder-path");
			});
	}

	@Test
	@DisplayName("PPOCR enabled=true 但缺 det-model-path：fail-fast 抛 MicaAiException")
	void ppocrEnabledButMissingRequiredShouldFailFast() {
		runner(PPOCRAutoConfiguration.class)
			.withPropertyValues("mica.ai.ppocr.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class)
					.hasMessageContaining("mica.ai.ppocr.det-model-path");
			});
	}

	@Test
	@DisplayName("Speaker enabled=true 但缺 model-path：fail-fast 抛 MicaAiException")
	void speakerEnabledButMissingRequiredShouldFailFast() {
		runner(SpeakerVerifierAutoConfiguration.class)
			.withPropertyValues("mica.ai.speaker.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class)
					.hasMessageContaining("mica.ai.speaker.model-path");
			});
	}

	@Test
	@DisplayName("Intent enabled=true 但缺 labels：fail-fast 抛 MicaAiException")
	void intentEnabledButMissingLabelsShouldFailFast() {
		String tmp = System.getProperty("java.io.tmpdir");
		runner(BertIntentAutoConfiguration.class)
			.withPropertyValues(
				"mica.ai.intent.enabled=true",
				"mica.ai.intent.model-path=" + tmp + "/dummy.onnx",
				"mica.ai.intent.vocab-path=" + tmp + "/vocab.txt"
			)
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class)
					.hasMessageContaining("mica.ai.intent.labels");
			});
	}

	@Test
	@DisplayName("Face enabled=true 但缺 det-model-path：fail-fast 抛 MicaAiException")
	void faceEnabledButMissingRequiredShouldFailFast() {
		runner(FaceAutoConfiguration.class)
			.withPropertyValues("mica.ai.face.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class)
					.hasMessageContaining("mica.ai.face.det-model-path");
			});
	}

	@Test
	@DisplayName("OpenCVNativeLoader：PPOCR 关闭时仍注册（独立 AutoConfiguration，与 enabled 解耦）")
	void openCVNativeLoaderRegisteredRegardlessOfEnabledFlag() {
		runner(OpenCVNativeLoader.class, PPOCRAutoConfiguration.class)
			.withPropertyValues("mica.ai.ppocr.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(OpenCVNativeLoader.OpenCVNativeBootstrap.class);
				assertNoEngine(ctx, PPOcrV6Engine.class);
			});
	}

	@Test
	@DisplayName("OpenCVNativeLoader：注册后 OpenCV Core.VERSION 可用（native 已成功 load）")
	void openCVNativeLoaderSuccessfullyLoadsNativeLibrary() {
		runner(OpenCVNativeLoader.class)
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(OpenCVNativeLoader.OpenCVNativeBootstrap.class);
				org.opencv.core.Core.VERSION.toString();
			});
	}

	@Test
	@DisplayName("OpenCVNativeLoader：PPOCR 启用 + 缺必填时仍按预期 fail-fast（@AutoConfigureBefore 不会屏蔽 PPOCRAutoConfiguration 校验）")
	void openCVNativeLoaderDoesNotMaskPpocrFailFast() {
		runner(OpenCVNativeLoader.class, PPOCRAutoConfiguration.class)
			.withPropertyValues("mica.ai.ppocr.enabled=true")
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class)
					.hasMessageContaining("mica.ai.ppocr.det-model-path");
			});
	}

	private static void assertNoEngine(org.springframework.boot.test.context.assertj.AssertableApplicationContext ctx, Class<?> type) {
		assertThat(ctx.getBeanNamesForType(type))
			.as("Bean %s 在 enabled=false 时不应存在", type.getSimpleName())
			.isEmpty();
	}
}
