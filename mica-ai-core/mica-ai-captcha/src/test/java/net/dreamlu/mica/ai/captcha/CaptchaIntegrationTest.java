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
package net.dreamlu.mica.ai.captcha;

import net.dreamlu.mica.ai.captcha.config.CaptchaConfig;
import net.dreamlu.mica.ai.captcha.config.CaptchaNormalization;
import net.dreamlu.mica.ai.captcha.model.CaptchaResult;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 集成测试：加载真实验证码 ONNX（{@code model-tools/captcha/models/ddddocr.onnx}）
 * 跑一遍完整链路（解码 → 等比缩放 → 灰度化 → 归一化 → ONNX → softmax → CTC 解码）。
 *
 * <p>覆盖四类易错点：
 * <ul>
 *   <li><b>I/O 契约</b>：输入名 {@code input1}、形状 {@code [1,1,64,W]}、
 *       输出运行时形状 {@code [seq,1,8210]}（量化模型声明的 {@code [1,seqlen]} 是错的）</li>
 *   <li><b>字符集对齐</b>：字符集类别数必须等于模型输出类别数，否则识别结果是乱码</li>
 *   <li><b>CTC blank 索引</b>：索引 0 必须是 blank，否则每个字符都会被重复输出</li>
 *   <li><b>归一化</b>：{@code DIV_255} 与 {@code MEAN_0_5} 在动态量化模型上应给出同一结果</li>
 * </ul>
 *
 * <p><b>模型缺失时整个测试类跳过</b>（{@code Assumptions}）：模型 12.98MB 在 &lt;50MB 约定内
 * 随仓库分发，但新克隆仓库若未拉取大文件仍可能缺失。
 *
 * <p>用外置模型跑同一套断言：
 * <pre>
 * mvn -pl mica-ai-core/mica-ai-captcha -am test \
 *     -Dmica.ai.captcha.externalModel=/path/to/model.onnx \
 *     -Dmica.ai.captcha.externalCharset=/path/to/charset.txt
 * </pre>
 */
class CaptchaIntegrationTest {

	private static final String MODEL_RELATIVE = "model-tools/captcha/models/ddddocr.onnx";
	private static final String CHARSET_RELATIVE = "model-tools/captcha/models/charset.txt";
	private static final String IMAGE_RESOURCE = "/captcha_shtp.png";
	private static final String EXPECTED_TEXT = "SHTP";

	private static CaptchaEngine engine;
	private static Path modelPath;
	private static byte[] imageBytes;

	@BeforeAll
	static void setUp() throws Exception {
		String externalModel = System.getProperty("mica.ai.captcha.externalModel");
		String externalCharset = System.getProperty("mica.ai.captcha.externalCharset");
		if (externalModel != null && !externalModel.trim().isEmpty()) {
			modelPath = Paths.get(externalModel);
			Assumptions.assumeTrue(Files.exists(modelPath),
				"外置模型不存在: " + externalModel + "，跳过真模型集成测试");
		} else {
			modelPath = locateRepoFile(MODEL_RELATIVE);
			Assumptions.assumeTrue(modelPath != null && Files.exists(modelPath),
				"未找到 " + MODEL_RELATIVE + "，跳过真模型集成测试");
		}
		Path charsetPath = externalCharset != null && !externalCharset.trim().isEmpty()
			? Paths.get(externalCharset)
			: locateRepoFile(CHARSET_RELATIVE);
		Assumptions.assumeTrue(charsetPath != null && Files.exists(charsetPath),
			"未找到字符集文件 " + CHARSET_RELATIVE + "，跳过真模型集成测试");

		imageBytes = readResource(IMAGE_RESOURCE);
		engine = CaptchaEngine.create(CaptchaConfig.builder()
			.modelPath(modelPath.toString())
			.charsetPath(charsetPath.toString())
			.build());
	}

	@AfterAll
	static void tearDown() {
		if (engine != null) {
			engine.close();
		}
	}

	@Test
	void recognizeShouldReturnExpectedText() {
		String text = engine.recognize(imageBytes);

		System.out.println("=== mica-ai-captcha 真模型集成测试 ===");
		System.out.println("识别结果: " + text);

		assertThat(text).isEqualTo(EXPECTED_TEXT);
	}

	@Test
	void recognizeDetailedShouldReportHighConfidence() {
		CaptchaResult result = engine.recognizeDetailed(imageBytes);

		System.out.printf("文本=%s 置信度=%.4f 逐字符=%s%n",
			result.getText(), result.getConfidence(),
			java.util.Arrays.toString(result.getCharConfidences()));

		assertThat(result.getText()).isEqualTo(EXPECTED_TEXT);
		assertThat(result.isEmpty()).isFalse();
		assertThat(result.getCharConfidences()).hasSize(EXPECTED_TEXT.length());
		assertThat(result.getConfidence()).isGreaterThan(0.9f);
		assertThat(result.isConfident(0.5f)).isTrue();
	}

	@Test
	void recognizeBase64ShouldMatchBytesResult() {
		String base64 = "data:image/png;base64,"
			+ Base64.getEncoder().encodeToString(imageBytes);

		assertThat(engine.recognizeBase64(base64)).isEqualTo(EXPECTED_TEXT);
	}

	@Test
	void bothNormalizationsShouldYieldSameText() {
		CaptchaConfig meanConfig = CaptchaConfig.builder()
			.modelPath(modelPath.toString())
			.charsetPath(engine.getConfig().getCharsetPath())
			.normalization(CaptchaNormalization.MEAN_0_5)
			.build();
		try (CaptchaEngine meanEngine = CaptchaEngine.create(meanConfig)) {
			assertThat(meanEngine.recognize(imageBytes)).isEqualTo(EXPECTED_TEXT);
		}
	}

	@Test
	void inputHeightMismatchShouldFailFast() {
		CaptchaConfig config = CaptchaConfig.builder()
			.modelPath(modelPath.toString())
			.charsetPath(engine.getConfig().getCharsetPath())
			.inputHeight(32)
			.build();
		assertThatThrownBy(() -> CaptchaEngine.create(config))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputHeight");
	}

	@Test
	void validConfigShouldPassValidation() {
		CaptchaConfig config = CaptchaConfig.builder()
			.modelPath(modelPath.toString())
			.charsetPath(engine.getConfig().getCharsetPath())
			.build();
		config.validate();
	}

	private static byte[] readResource(String name) throws Exception {
		try (java.io.InputStream in = CaptchaIntegrationTest.class.getResourceAsStream(name)) {
			assertThat(in).as("测试资源必须存在: " + name).isNotNull();
			java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
			byte[] buffer = new byte[4096];
			int read;
			while ((read = in.read(buffer)) != -1) {
				out.write(buffer, 0, read);
			}
			return out.toByteArray();
		}
	}

	/**
	 * 从工作目录向上找到仓库根（以根 {@code pom.xml} 为标志），再拼出相对路径。
	 * 找不到仓库根时断言失败；文件不存在时返回 {@code null}，由调用方决定跳过还是失败。
	 */
	private static Path locateRepoFile(String relative) {
		Path root = Paths.get("").toAbsolutePath();
		while (root != null && !Files.exists(root.resolve("pom.xml"))) {
			root = root.getParent();
		}
		assertThat(root).as("找不到仓库根 pom.xml").isNotNull();
		Path target = root.resolve(relative);
		return Files.exists(target) ? target : null;
	}
}
