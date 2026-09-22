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
package net.dreamlu.mica.ai.example;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.autoconfigure.FaceAutoConfiguration;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.matting.MattingEngine;
import net.dreamlu.mica.ai.matting.autoconfigure.MattingAutoConfiguration;
import net.dreamlu.mica.ai.textline.TextLineEngine;
import net.dreamlu.mica.ai.textline.autoconfigure.TextLineAutoConfiguration;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * mica-ai-example 集成测试：验证 Spring Boot Starter 的自动装配行为。
 *
 * <p>统一使用轻量级 {@link ApplicationContextRunner}（非 {@code @SpringBootTest}），
 * 规避 Surefire fork 问题。
 */
class ExampleApplicationContextTest {

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

	@Test
	@DisplayName("Matting 关闭时不应创建 MattingEngine，且不会因缺模型而启动失败")
	void mattingDisabledShouldNotCreateEngine() {
		runner(MattingAutoConfiguration.class)
			.withPropertyValues("mica.ai.matting.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(MattingEngine.class);
			});
	}

	@Test
	@DisplayName("Matting 指向真实 u2netp 模型时应装配出可用的 MattingEngine")
	void mattingShouldWireEngineAgainstRealModel() {
		Path model = locateRepoFile("model-tools/matting/models/u2netp.onnx");
		if (model == null) {
			// 模型未入库的场景下跳过，不阻塞新克隆仓库的构建
			return;
		}
		runner(MattingAutoConfiguration.class)
			.withPropertyValues("mica.ai.matting.model-path=" + model.toAbsolutePath())
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(MattingEngine.class);
				MattingEngine engine = ctx.getBean(MattingEngine.class);
				assertThat(engine.modelInputSize()).isEqualTo(320);
				assertThat(engine.outputCount()).isEqualTo(7);
			});
	}

	@Test
	@DisplayName("TextLine 关闭时不应创建 TextLineEngine，且不会因缺模型而启动失败")
	void textlineDisabledShouldNotCreateEngine() {
		runner(TextLineAutoConfiguration.class)
			.withPropertyValues("mica.ai.textline.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(TextLineEngine.class);
			});
	}

	@Test
	@DisplayName("TextLine 指向真实 PP-LCNet 模型时应装配出可用的 TextLineEngine")
	void textlineShouldWireEngineAgainstRealModel() {
		Path model = locateRepoFile(
			"model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx");
		if (model == null) {
			// 模型未入库的场景下跳过，不阻塞新克隆仓库的构建
			return;
		}
		runner(TextLineAutoConfiguration.class)
			.withPropertyValues("mica.ai.textline.model-path=" + model.toAbsolutePath())
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(TextLineEngine.class);
				TextLineEngine engine = ctx.getBean(TextLineEngine.class);
				assertThat(engine.modelInputWidth()).isEqualTo(160);
				assertThat(engine.modelInputHeight()).isEqualTo(80);
				assertThat(engine.classCount()).isEqualTo(2);
			});
	}

	/**
	 * 从当前工作目录向上查找仓库根，定位入库模型。
	 */
	private static Path locateRepoFile(String relative) {
		Path dir = Paths.get("").toAbsolutePath();
		for (int i = 0; i < 6 && dir != null; i++) {
			Path candidate = dir.resolve(relative);
			if (Files.exists(candidate)) {
				return candidate;
			}
			dir = dir.getParent();
		}
		return null;
	}

	@SpringBootConfiguration
	static class EmptyConfig {
	}
}
