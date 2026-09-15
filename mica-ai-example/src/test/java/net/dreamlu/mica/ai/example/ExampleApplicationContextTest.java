/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.autoconfigure.FaceAutoConfiguration;
import net.dreamlu.mica.ai.face.model.ModelManager;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * mica-ai-example 集成测试：验证 Spring Boot Starter 的自动装配行为。
 *
 * <p>统一使用轻量级 {@link ApplicationContextRunner}（非 {@code @SpringBootTest}），
 * 规避 Surefire fork 问题。
 */
class ExampleApplicationContextTest {

	@SpringBootConfiguration
	static class EmptyConfig {
	}

	private static ApplicationContextRunner runner(Class<?>... autoConfigs) {
		return new ApplicationContextRunner()
			.withUserConfiguration(EmptyConfig.class)
			.withConfiguration(AutoConfigurations.of(autoConfigs));
	}

	@Test
	@DisplayName("Face Starter 加载且缺少必填模型路径时 fail-fast 抛 MicaAiException")
	void faceEnabledButMissingRequiredShouldFailFast() {
		runner(FaceAutoConfiguration.class)
			.withPropertyValues(
				"mica.ai.face.model.detection.path=",
				"mica.ai.face.model.recognition.path=",
				"mica.ai.face.liveness.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class);
			});
	}

	@Test
	@DisplayName("ModelManager 类可在 face 模块解析（仅校验类型，不触发加载）")
	void modelManagerClassIsResolvable() {
		// 仅验证 face 模块的 ModelManager 类可被加载，绕过 Spring / 原生库
		Class<?> type = ModelManager.class;
		assertThat(type.getName()).isEqualTo(
			"net.dreamlu.mica.ai.face.model.ModelManager");
	}
}