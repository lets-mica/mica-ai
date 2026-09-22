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
package net.dreamlu.mica.ai.textline;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.util.IOUtil;
import net.dreamlu.mica.ai.textline.config.TextLineChannelOrder;
import net.dreamlu.mica.ai.textline.config.TextLineConfig;
import net.dreamlu.mica.ai.textline.model.TextLineOrientation;
import net.dreamlu.mica.ai.textline.model.TextLineOrientationResult;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 集成测试：加载仓库内真实 PP-LCNet_x1_0_textline_ori ONNX
 * （{@code model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx}，6.46MB）
 * 跑一遍完整链路（resize → 归一化 → ONNX → softmax → 阈值判定）。
 *
 * <p>覆盖四类易错点：
 * <ul>
 *   <li><b>I/O 契约</b>：输入名 {@code x}、形状 {@code [1,3,80,160]}、2 类输出，换模型立刻报警</li>
 *   <li><b>类别语义</b>：索引 1 必须是 180 度（官方 {@code label_list: [0_degree, 180_degree]}），
 *       写反会导致「把正的转成倒的」，比不判还糟</li>
 *   <li><b>宽高不得写反</b>：{@code Imgproc.resize} 的 {@code Size} 是 (宽, 高)，
 *       窄长文本行上写反会被拉成竖排，方向判定随之失真</li>
 *   <li><b>通道顺序不敏感</b>：本任务判定的是结构朝向而非颜色，BGR/RGB 应给出相同结论</li>
 * </ul>
 *
 * <p><b>模型缺失时整个测试类跳过</b>（{@code Assumptions}）：模型 6.46MB 在 &lt;50MB 约定内
 * 随仓库分发，但新克隆仓库若未拉取大文件仍可能缺失，跳过时报告会打印提示。
 *
 * <p>OpenCV 原生库由 openpnp 的 {@code nu.pattern.OpenCV.loadShared()} 在静态块中加载。
 */
class TextLineIntegrationTest {

	static {
		nu.pattern.OpenCV.loadShared();
	}

	private static final String MODEL_RELATIVE =
		"model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx";

	private static TextLineEngine engine;
	private static Path modelPath;

	@BeforeAll
	static void setUp() {
		modelPath = locateRepoFile(MODEL_RELATIVE);
		Assumptions.assumeTrue(modelPath != null && Files.exists(modelPath),
			"未找到 " + MODEL_RELATIVE + "，跳过真模型集成测试");
		engine = TextLineEngine.create(TextLineConfig.builder()
			.modelPath(modelPath.toString())
			.build());
	}

	@AfterAll
	static void tearDown() {
		if (engine != null) {
			engine.close();
		}
	}

	// ---------------------------------------------------------------- I/O 契约

	@Test
	void modelContractShouldMatchLcnetTextLineOri() {
		try (TextLineEngine probe = TextLineEngine.create(TextLineConfig.builder()
			.modelPath(modelPath.toString())
			.build())) {
			assertThat(probe.modelInputWidth()).as("官方输入宽应为 160").isEqualTo(160);
			assertThat(probe.modelInputHeight()).as("官方输入高应为 80").isEqualTo(80);
			assertThat(probe.classCount()).as("官方为 2 类（0_degree / 180_degree）").isEqualTo(2);
		}
	}

	@Test
	void inputSizeMismatchShouldFailFast() {
		TextLineConfig config = TextLineConfig.builder()
			.modelPath(modelPath.toString())
			.inputWidth(224)
			.inputHeight(224)
			.build();
		assertThatThrownBy(() -> TextLineEngine.create(config))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("input");
	}

	/**
	 * 宽高写反必须快速失败——它是最隐蔽的坑：不报错、只是判定质量悄悄变差。
	 */
	@Test
	void swappedWidthHeightShouldFailFast() {
		TextLineConfig swapped = TextLineConfig.builder()
			.modelPath(modelPath.toString())
			.inputWidth(80)
			.inputHeight(160)
			.build();
		assertThatThrownBy(() -> TextLineEngine.create(swapped))
			.isInstanceOf(MicaAiException.class);
	}

	// ---------------------------------------------------------------- 方向判定正确性

	@Test
	void uprightLineShouldNotBeUpsideDown() throws Exception {
		TextLineOrientationResult r = engine.classifyBytes(readResource("/upright.png"));
		System.out.printf("=== upright.png: %s score=%.4f ===%n", r.getOrientation(), r.getScore());
		assertThat(r.getOrientation()).isEqualTo(TextLineOrientation.DEGREE_0);
		assertThat(r.isUpsideDown()).isFalse();
		assertThat(r.getScore()).isGreaterThan(0.5f);
	}

	@Test
	void flippedLineShouldBeDetectedAsUpsideDown() throws Exception {
		TextLineOrientationResult r = engine.classifyBytes(readResource("/flipped.png"));
		System.out.printf("=== flipped.png: %s score=%.4f ===%n", r.getOrientation(), r.getScore());
		assertThat(r.getOrientation()).as("倒置行必须判为 180 度（索引 1）")
			.isEqualTo(TextLineOrientation.DEGREE_180);
		assertThat(r.isUpsideDown()).isTrue();
		assertThat(r.angle()).as("转正角度应为 180").isEqualTo(180);
	}

	/**
	 * 类别索引语义必须与官方一致：索引 1 = 180 度。
	 *
	 * <p>这条断言是「方向判反」的兜底——若把 label_list 顺序搞反，
	 * 正的会被判成倒的，后果比不判更严重。
	 */
	@Test
	void classIndexSemanticsShouldMatchOfficialLabelList() {
		assertThat(TextLineOrientation.fromClassIndex(0)).isEqualTo(TextLineOrientation.DEGREE_0);
		assertThat(TextLineOrientation.fromClassIndex(1)).isEqualTo(TextLineOrientation.DEGREE_180);
		assertThat(TextLineOrientation.DEGREE_0.getLabel()).isEqualTo("0_degree");
		assertThat(TextLineOrientation.DEGREE_180.getLabel()).isEqualTo("180_degree");
		assertThat(TextLineOrientation.DEGREE_0.getAngle()).isEqualTo(0);
		assertThat(TextLineOrientation.DEGREE_180.getAngle()).isEqualTo(180);
	}

	/**
	 * 前景/背景极性不应影响判定：白字深底与黑字白底的结论必须一致。
	 */
	@Test
	void polarityShouldNotAffectVerdict() throws Exception {
		TextLineOrientationResult upright = engine.classifyBytes(readResource("/inverted.png"));
		TextLineOrientationResult flipped = engine.classifyBytes(readResource("/inverted_flipped.png"));
		System.out.printf("=== inverted: upright=%s(%.4f) flipped=%s(%.4f) ===%n",
			upright.getOrientation(), upright.getScore(),
			flipped.getOrientation(), flipped.getScore());
		assertThat(upright.getOrientation()).isEqualTo(TextLineOrientation.DEGREE_0);
		assertThat(flipped.getOrientation()).isEqualTo(TextLineOrientation.DEGREE_180);
	}

	/**
	 * 宽高写反会破坏判定：同一批图在宽高互换的配置下应当「快速失败」，
	 * 已由 {@link #swappedWidthHeightShouldFailFast()} 覆盖。
	 *
	 * <p>这里进一步验证<b>窄长文本行</b>（700x60）在正确配置下方向判定仍然正确——
	 * 这类输入的宽高比 11.7:1，最容易被 resize 参数错位影响。
	 */
	@Test
	void wideLineShouldStillBeClassifiedCorrectly() throws Exception {
		TextLineOrientationResult upright = engine.classifyBytes(readResource("/wide.png"));
		TextLineOrientationResult flipped = engine.classifyBytes(readResource("/wide_flipped.png"));
		System.out.printf("=== wide: upright=%s(%.4f) flipped=%s(%.4f) ===%n",
			upright.getOrientation(), upright.getScore(),
			flipped.getOrientation(), flipped.getScore());
		assertThat(upright.getOrientation()).isEqualTo(TextLineOrientation.DEGREE_0);
		assertThat(flipped.getOrientation()).isEqualTo(TextLineOrientation.DEGREE_180);
	}

	/**
	 * 官方真实样例（PaddleX {@code textline_rot180_demo.jpg}）：
	 * 判定必须正确，<b>并锁住真实数据的置信度水平</b>。
	 *
	 * <p>⚠️ 这条断言的价值在于记录一个实测事实：<b>真实图片上的置信度会明显低于合成图片</b>。
	 * 合成文本行（干净、大字号、无噪）在本模型上会饱和到 {@code logits=[+1.0, +0.0]}，
	 * 概率恒为 {@code 0.7311}；而真实扫描件上倒置态仅得 {@code p(180)≈0.5772}，
	 * 已经逼近默认阈值 {@code 0.5}。
	 *
	 * <p>因此本测试用 {@code 0.55} 这种「真实数据可达到」的下界做断言，
	 * 而不是照抄合成图上的 0.73 —— 后者会让测试无法反映真实场景的退化风险。
	 */
	@Test
	void realWorldSampleShouldBeClassifiedAndScoreShouldBeRealistic() throws Exception {
		TextLineOrientationResult r = engine.classifyBytes(readResource("/real_flipped.jpg"));
		System.out.printf("=== real_flipped.jpg: %s score=%.4f ===%n", r.getOrientation(), r.getScore());
		assertThat(r.getOrientation()).as("官方 180 度样例必须判为倒置")
			.isEqualTo(TextLineOrientation.DEGREE_180);
		assertThat(r.getScore()).as("真实数据置信度应高于阈值，但明显低于合成图的 0.7311")
			.isBetween(0.55f, 0.72f);

		// 转正后应稳定回到 0 度
		byte[] rotated = engine.uprightBytes(readResource("/real_flipped.jpg"));
		assertThat(engine.classifyBytes(rotated).getOrientation())
			.as("官方样例转正后应判为 0 度")
			.isEqualTo(TextLineOrientation.DEGREE_0);
	}

	// ---------------------------------------------------------------- 通道顺序

	/**
	 * 通道顺序不影响结论：本任务判定结构朝向，BGR 与 RGB 应给出相同方向。
	 *
	 * <p>该断言同时锁住「未来若有人误以为必须 RGB」的错误改动——只要方向依旧一致，
	 * 说明这个开关在默认模型上是安全的。
	 */
	@Test
	void channelOrderShouldNotChangeVerdict() throws Exception {
		byte[] uprightBytes = readResource("/upright.png");
		byte[] flippedBytes = readResource("/flipped.png");
		try (TextLineEngine rgbEngine = TextLineEngine.create(TextLineConfig.builder()
			.modelPath(modelPath.toString())
			.channelOrder(TextLineChannelOrder.RGB)
			.build())) {
			assertThat(rgbEngine.classifyBytes(uprightBytes).getOrientation())
				.isEqualTo(engine.classifyBytes(uprightBytes).getOrientation());
			assertThat(rgbEngine.classifyBytes(flippedBytes).getOrientation())
				.isEqualTo(engine.classifyBytes(flippedBytes).getOrientation());
		}
	}

	// ---------------------------------------------------------------- 转正输出

	@Test
	void uprightBytesShouldRotateFlippedLine() throws Exception {
		byte[] flipped = readResource("/flipped.png");
		byte[] rotated = engine.uprightBytes(flipped);
		Mat decoded = decode(rotated);
		try {
			assertThat(decoded.cols()).isEqualTo(400);
			assertThat(decoded.rows()).isEqualTo(80);
		} finally {
			decoded.release();
		}
		// 转正后的图再判一次，必须变成 0 度 —— 形成闭环
		TextLineOrientationResult after = engine.classifyBytes(rotated);
		System.out.printf("=== 转正后复检: %s score=%.4f ===%n", after.getOrientation(), after.getScore());
		assertThat(after.getOrientation()).as("转正后应判为 0 度").isEqualTo(TextLineOrientation.DEGREE_0);
	}

	@Test
	void uprightBytesShouldReturnInputUntouchedWhenAlreadyUpright() throws Exception {
		byte[] upright = readResource("/upright.png");
		byte[] out = engine.uprightBytes(upright);
		assertThat(out).as("方向正常时应原样返回输入字节，不做无谓重编码").isSameAs(upright);
	}

	// ---------------------------------------------------------------- 阈值策略

	/**
	 * 阈值必须真的生效：把阈值拉到 1.0 时，任何行都不应被判为倒置。
	 */
	@Test
	void upsideDownThresholdShouldActuallyBeApplied() throws Exception {
		byte[] flipped = readResource("/flipped.png");
		try (TextLineEngine strict = TextLineEngine.create(TextLineConfig.builder()
			.modelPath(modelPath.toString())
			.upsideDownThreshold(1.0f)
			.build())) {
			TextLineOrientationResult r = strict.classifyBytes(flipped);
			System.out.printf("=== threshold=1.0 on flipped.png: %s score=%.4f ===%n",
				r.getOrientation(), r.getScore());
			assertThat(r.isUpsideDown()).as("阈值 1.0 时不应判为倒置").isFalse();
		}
		assertThat(engine.classifyBytes(flipped).isUpsideDown())
			.as("默认阈值下同一张图应判为倒置").isTrue();
	}

	// ---------------------------------------------------------------- 兜底

	@Test
	void pathInputShouldWorkAndMissingFileShouldThrow() throws Exception {
		byte[] bytes = readResource("/upright.png");
		Path tmp = Files.createTempFile("mica-ai-textline-", ".png");
		try {
			Files.write(tmp, bytes);
			assertThat(engine.classifyPath(tmp.toString()).getOrientation())
				.isEqualTo(TextLineOrientation.DEGREE_0);
		} finally {
			Files.deleteIfExists(tmp);
		}
		assertThatThrownBy(() -> engine.classifyPath("no-such-file-xyz.png"))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("文件不存在");
	}

	@Test
	void nullOrEmptyInputShouldBeHandledGracefully() {
		assertThat(engine.classify((Mat) null)).isNull();
		assertThat(engine.classify(new Mat())).isNull();
	}

	/**
	 * 外置模型接入：指向 {@code model-path} 的同任务模型都应能加载
	 * （如轻量版 {@code PP-LCNet_x0_25_textline_ori}）。
	 */
	@Test
	void externalModelShouldBeLoadableWhenProvided() {
		String external = System.getProperty("mica.ai.textline.externalModel");
		Assumptions.assumeTrue(external != null && Files.exists(Paths.get(external)),
			"未通过 -Dmica.ai.textline.externalModel 提供外置模型，跳过");
		try (TextLineEngine externalEngine = TextLineEngine.create(TextLineConfig.builder()
			.modelPath(external)
			.build())) {
			assertThat(externalEngine.classCount()).isEqualTo(2);
		}
	}

	// ---------------------------------------------------------------- 辅助

	private static Mat decode(byte[] bytes) {
		MatOfByte mob = new MatOfByte(bytes);
		try {
			Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_UNCHANGED);
			assertThat(mat.empty()).as("PNG 必须可解码").isFalse();
			return mat;
		} finally {
			mob.release();
		}
	}

	private static byte[] readResource(String name) throws Exception {
		try (InputStream in = TextLineIntegrationTest.class.getResourceAsStream(name)) {
			assertThat(in).as("测试资源必须存在: " + name).isNotNull();
			return IOUtil.readAllBytes(in);
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
