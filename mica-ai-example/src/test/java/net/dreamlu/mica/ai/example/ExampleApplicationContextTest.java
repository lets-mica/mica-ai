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

import net.dreamlu.mica.ai.captcha.CaptchaEngine;
import net.dreamlu.mica.ai.captcha.autoconfigure.CaptchaAutoConfiguration;
import net.dreamlu.mica.ai.classify.autoconfigure.ClassifyAutoConfiguration;
import net.dreamlu.mica.ai.classify.core.ClassifyEngine;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.detect.autoconfigure.DetectAutoConfiguration;
import net.dreamlu.mica.ai.detect.core.DetectEngine;
import net.dreamlu.mica.ai.docalign.autoconfigure.DocAlignAutoConfiguration;
import net.dreamlu.mica.ai.docalign.core.DocAlignEngine;
import net.dreamlu.mica.ai.enhance.autoconfigure.EnhanceAutoConfiguration;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.face.autoconfigure.FaceAutoConfiguration;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.humanseg.autoconfigure.HumanSegAutoConfiguration;
import net.dreamlu.mica.ai.humanseg.core.HumanSegEngine;
import net.dreamlu.mica.ai.matting.autoconfigure.MattingAutoConfiguration;
import net.dreamlu.mica.ai.matting.core.MattingEngine;
import net.dreamlu.mica.ai.textline.autoconfigure.TextLineAutoConfiguration;
import net.dreamlu.mica.ai.textline.core.TextLineEngine;
import net.dreamlu.mica.ai.track.autoconfigure.TrackAutoConfiguration;
import net.dreamlu.mica.ai.track.core.ByteTracker;
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

	@Test
	@DisplayName("Captcha 关闭时不应创建 CaptchaEngine，且不会因缺模型而启动失败")
	void captchaDisabledShouldNotCreateEngine() {
		runner(CaptchaAutoConfiguration.class)
			.withPropertyValues("mica.ai.captcha.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(CaptchaEngine.class);
			});
	}

	@Test
	@DisplayName("Captcha 缺少必填模型路径时应 fail-fast 抛 MicaAiException（模型外置配置）")
	void captchaEnabledButMissingRequiredShouldFailFast() {
		runner(CaptchaAutoConfiguration.class)
			.withPropertyValues(
				"mica.ai.captcha.model-path=",
				"mica.ai.captcha.charset-path=")
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class);
			});
	}

	@Test
	@DisplayName("Captcha 指向真实 ddddocr 模型时应装配出可用的 CaptchaEngine")
	void captchaShouldWireEngineAgainstRealModel() {
		Path model = locateRepoFile("model-tools/captcha/models/ddddocr.onnx");
		Path charset = locateRepoFile("model-tools/captcha/models/charset.txt");
		if (model == null || charset == null) {
			// 模型未入库的场景下跳过，不阻塞新克隆仓库的构建
			return;
		}
		runner(CaptchaAutoConfiguration.class)
			.withPropertyValues(
				"mica.ai.captcha.model-path=" + model.toAbsolutePath(),
				"mica.ai.captcha.charset-path=" + charset.toAbsolutePath())
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(CaptchaEngine.class);
				CaptchaEngine engine = ctx.getBean(CaptchaEngine.class);
				assertThat(engine.modelInputHeight()).isEqualTo(64);
				// 8209 个字符 + 1 个 blank
				assertThat(engine.charsetSize()).isEqualTo(8210);
			});
	}

	@Test
	@DisplayName("Detect 关闭时不应创建 DetectEngine，且不会因缺模型而启动失败")
	void detectDisabledShouldNotCreateEngine() {
		runner(DetectAutoConfiguration.class)
			.withPropertyValues("mica.ai.detect.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(DetectEngine.class);
			});
	}

	@Test
	@DisplayName("Detect 指向真实 NanoDet-Plus 模型时应装配出可用的 DetectEngine")
	void detectShouldWireEngineAgainstRealModel() {
		Path model = locateRepoFile(
			"model-tools/detect/models/object_detection_nanodet_2022nov.onnx");
		if (model == null) {
			// 模型未入库的场景下跳过，不阻塞新克隆仓库的构建
			return;
		}
		runner(DetectAutoConfiguration.class)
			.withPropertyValues("mica.ai.detect.model-path=" + model.toAbsolutePath())
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(DetectEngine.class);
				DetectEngine engine = ctx.getBean(DetectEngine.class);
				assertThat(engine.modelInputSize()).isEqualTo(416);
				assertThat(engine.outputCount()).isEqualTo(6);
				assertThat(engine.getLabels()).hasSize(80);
			});
	}

	@Test
	@DisplayName("Track 关闭时不应创建 ByteTracker")
	void trackDisabledShouldNotCreateTracker() {
		runner(TrackAutoConfiguration.class)
			.withPropertyValues("mica.ai.track.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(ByteTracker.class);
			});
	}

	@Test
	@DisplayName("Track 开启时应装配出 ByteTracker，且阈值取 mica.ai.track 配置")
	void trackShouldWireTracker() {
		runner(TrackAutoConfiguration.class)
			.withPropertyValues("mica.ai.track.high-threshold=0.7")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(ByteTracker.class);
				ByteTracker tracker = ctx.getBean(ByteTracker.class);
				assertThat(tracker.getConfig().getHighThreshold()).isEqualTo(0.7f);
				assertThat(tracker.trackCount()).isZero();
			});
	}

	@Test
	@DisplayName("HumanSeg 关闭时不应创建 HumanSegEngine，且不会因缺模型而启动失败")
	void humansegDisabledShouldNotCreateEngine() {
		runner(HumanSegAutoConfiguration.class)
			.withPropertyValues("mica.ai.humanseg.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(HumanSegEngine.class);
			});
	}

	@Test
	@DisplayName("HumanSeg 指向真实 PP-HumanSeg 模型时应装配出可用的 HumanSegEngine")
	void humansegShouldWireEngineAgainstRealModel() {
		Path model = locateRepoFile(
			"model-tools/humanseg/models/human_segmentation_pphumanseg_2023mar.onnx");
		if (model == null) {
			// 模型未入库的场景下跳过，不阻塞新克隆仓库的构建
			return;
		}
		runner(HumanSegAutoConfiguration.class)
			.withPropertyValues("mica.ai.humanseg.model-path=" + model.toAbsolutePath())
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(HumanSegEngine.class);
				HumanSegEngine engine = ctx.getBean(HumanSegEngine.class);
				assertThat(engine.modelInputWidth()).isEqualTo(192);
				assertThat(engine.modelInputHeight()).isEqualTo(192);
				assertThat(engine.outputCount()).isEqualTo(1);
			});
	}

	@Test
	@DisplayName("DocAlign 关闭时不应创建 DocAlignEngine，且不会因缺模型而启动失败")
	void docalignDisabledShouldNotCreateEngine() {
		runner(DocAlignAutoConfiguration.class)
			.withPropertyValues("mica.ai.docalign.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(DocAlignEngine.class);
			});
	}

	@Test
	@DisplayName("DocAlign 指向真实 DocAligner 模型时应装配出可用的 DocAlignEngine")
	void docalignShouldWireEngineAgainstRealModel() {
		Path model = locateRepoFile(
			"model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx");
		if (model == null) {
			// 模型未入库的场景下跳过，不阻塞新克隆仓库的构建
			return;
		}
		runner(DocAlignAutoConfiguration.class)
			.withPropertyValues("mica.ai.docalign.model-path=" + model.toAbsolutePath())
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(DocAlignEngine.class);
				DocAlignEngine engine = ctx.getBean(DocAlignEngine.class);
				assertThat(engine.modelInputWidth()).isEqualTo(256);
				assertThat(engine.modelInputHeight()).isEqualTo(256);
				assertThat(engine.outputCount()).isEqualTo(1);
			});
	}

	@Test
	@DisplayName("Enhance 关闭时不应创建 EnhanceEngine，且不会因缺模型而启动失败")
	void enhanceDisabledShouldNotCreateEngine() {
		runner(EnhanceAutoConfiguration.class)
			.withPropertyValues("mica.ai.enhance.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(EnhanceEngine.class);
			});
	}

	@Test
	@DisplayName("Enhance 指向真实 Real-ESRGAN 模型时应装配出可用的 EnhanceEngine")
	void enhanceShouldWireEngineAgainstRealModel() {
		Path model = locateRepoFile("model-tools/enhance/models/realesr-general-x4v3.onnx");
		if (model == null) {
			// 模型未入库的场景下跳过，不阻塞新克隆仓库的构建
			return;
		}
		runner(EnhanceAutoConfiguration.class)
			.withPropertyValues("mica.ai.enhance.model-path=" + model.toAbsolutePath())
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(EnhanceEngine.class);
				EnhanceEngine engine = ctx.getBean(EnhanceEngine.class);
				assertThat(engine.getConfig().getScale()).isEqualTo(4);
				assertThat(engine.outputCount()).isEqualTo(1);
			});
	}

	@Test
	@DisplayName("Classify 关闭时不应创建 ClassifyEngine，且不会因缺模型而启动失败")
	void classifyDisabledShouldNotCreateEngine() {
		runner(ClassifyAutoConfiguration.class)
			.withPropertyValues("mica.ai.classify.enabled=false")
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).doesNotHaveBean(ClassifyEngine.class);
			});
	}

	@Test
	@DisplayName("Classify 缺少配套 label_list 时应 fail-fast 抛 MicaAiException（模型 + 标签成对配置）")
	void classifyEnabledButMissingLabelsShouldFailFast() {
		runner(ClassifyAutoConfiguration.class)
			.withPropertyValues(
				"mica.ai.classify.model-path=",
				"mica.ai.classify.labels-path=")
			.run(ctx -> {
				assertThat(ctx).hasFailed();
				assertThat(ctx.getStartupFailure())
					.hasRootCauseInstanceOf(MicaAiException.class);
			});
	}

	@Test
	@DisplayName("Classify 指向真实 PULC 模型 + label_list 时应装配出可用的 ClassifyEngine")
	void classifyShouldWireEngineAgainstRealModel() {
		Path model = locateRepoFile("model-tools/classify/models/text_image_orientation.onnx");
		Path labels = locateRepoFile("model-tools/classify/models/text_image_orientation_labels.txt");
		if (model == null || labels == null) {
			// 模型未入库的场景下跳过，不阻塞新克隆仓库的构建
			return;
		}
		runner(ClassifyAutoConfiguration.class)
			.withPropertyValues(
				"mica.ai.classify.model-path=" + model.toAbsolutePath(),
				"mica.ai.classify.labels-path=" + labels.toAbsolutePath())
			.run(ctx -> {
				assertThat(ctx).hasNotFailed();
				assertThat(ctx).hasSingleBean(ClassifyEngine.class);
				ClassifyEngine engine = ctx.getBean(ClassifyEngine.class);
				assertThat(engine.modelInputWidth()).isEqualTo(224);
				assertThat(engine.modelInputHeight()).isEqualTo(224);
				// 整图方向 0 / 90 / 180 / 270
				assertThat(engine.classCount()).isEqualTo(4);
			});
	}

	/**
	 * 从当前工作目录向上查找仓库根，定位入库模型。
	 */
	private static Path locateRepoFile(String relative) {		Path dir = Paths.get("").toAbsolutePath();
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
