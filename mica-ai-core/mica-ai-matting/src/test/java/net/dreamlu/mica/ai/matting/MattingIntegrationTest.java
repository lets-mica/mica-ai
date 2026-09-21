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
package net.dreamlu.mica.ai.matting;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.util.IOUtil;
import net.dreamlu.mica.ai.matting.config.MattingConfig;
import net.dreamlu.mica.ai.matting.config.MattingOutputSelect;
import net.dreamlu.mica.ai.matting.model.MattingResult;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
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
 * 集成测试：加载仓库内真实 u2netp ONNX（{@code model-tools/matting/models/u2netp.onnx}，4.4MB）
 * 跑一遍完整链路（resize → RGB/ImageNet 归一化 → ONNX → min-max → resize 回原尺寸 → 合成/编码）。
 *
 * <p>覆盖三类易错点：
 * <ul>
 *   <li><b>I/O 契约</b>：输入 {@code [1,3,320,320]}、7 个输出、d0 落在 [0,1]，换模型立刻报警</li>
 *   <li><b>掩码尺寸回填</b>：非方形 / 极端长宽比输入下 alpha 必须与原图同尺寸（OpenCV Size 是 (宽,高)）</li>
 *   <li><b>alpha 语义</b>：主体区域的 alpha 显著高于背景；min-max 拉伸确实作用于低动态范围输入</li>
 * </ul>
 *
 * <p><b>模型缺失时整个测试类跳过</b>（{@code Assumptions}）：模型 4.4MB 虽在 <50MB 约定内，
 * 但新克隆仓库若未拉取大文件仍可能缺失，跳过时报告会打印提示。
 *
 * <p>OpenCV 原生库由 openpnp 的 {@code nu.pattern.OpenCV.loadShared()} 在静态块中加载。
 */
class MattingIntegrationTest {

	static {
		nu.pattern.OpenCV.loadShared();
	}

	private static final String MODEL_RELATIVE = "model-tools/matting/models/u2netp.onnx";

	private static MattingEngine engine;
	private static Path modelPath;

	@BeforeAll
	static void setUp() {
		modelPath = locateRepoFile(MODEL_RELATIVE);
		Assumptions.assumeTrue(modelPath != null && Files.exists(modelPath),
			"未找到 " + MODEL_RELATIVE + "，跳过真模型集成测试");
		engine = MattingEngine.create(MattingConfig.builder()
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
	void modelContractShouldMatchU2netp() {
		// 直接探查 session，确认导出形态没变：1 输入 / 7 输出 / 320 输入边长
		MattingConfig config = MattingConfig.builder().modelPath(modelPath.toString()).build();
		try (MattingEngine probe = MattingEngine.create(config)) {
			assertThat(probe.getConfig().getInputSize()).isEqualTo(320);
			assertThat(probe.modelInputSize()).as("u2netp 输入边长应为 320").isEqualTo(320);
			assertThat(probe.outputCount()).as("u2netp 应输出 d0..d6 共 7 个节点").isEqualTo(7);
		}
	}

	@Test
	void inputSizeMismatchShouldFailFast() {
		MattingConfig config = MattingConfig.builder()
			.modelPath(modelPath.toString())
			.inputSize(256)
			.build();
		assertThatThrownBy(() -> MattingEngine.create(config))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("inputSize");
	}

	// ---------------------------------------------------------------- 掩码尺寸回填

	@Test
	void alphaShouldMatchOriginalSizeForNonSquareImage() throws Exception {
		byte[] bytes = readResource("/subject.png");
		int[] size = imageSize(bytes);
		try (MattingResult r = engine.matteBytes(bytes)) {
			assertThat(r.getAlpha().cols()).as("alpha 宽度必须等于原图宽度").isEqualTo(size[0]);
			assertThat(r.getAlpha().rows()).as("alpha 高度必须等于原图高度").isEqualTo(size[1]);
			assertThat(r.getWidth()).isEqualTo(size[0]);
			assertThat(r.getHeight()).isEqualTo(size[1]);
			assertThat(r.getAlpha().type()).isEqualTo(CvType.CV_32FC1);
		}
	}

	/**
	 * 极端长宽比（640×120）最容易暴露 Size(宽,高) 写反 —— 写反时 alpha 会变成 120×640，
	 * 与 Mat 的 rows/cols 语义直接冲突。
	 */
	@Test
	void alphaShouldMatchOriginalSizeForExtremeAspectRatio() throws Exception {
		byte[] bytes = readResource("/wide.png");
		int[] size = imageSize(bytes);
		assertThat(size[0]).as("测试图应为宽图").isGreaterThan(size[1]);
		try (MattingResult r = engine.matteBytes(bytes)) {
			assertThat(r.getAlpha().cols()).isEqualTo(640);
			assertThat(r.getAlpha().rows()).isEqualTo(120);
		}
	}

	// ---------------------------------------------------------------- alpha 语义

	@Test
	void alphaShouldBeInRangeAndHighlightSubject() throws Exception {
		byte[] bytes = readResource("/subject.png");
		try (MattingResult r = engine.matteBytes(bytes)) {
			Mat alpha = r.getAlpha();
			Core.MinMaxLocResult mm = Core.minMaxLoc(alpha);
			System.out.printf("=== subject.png alpha: min=%.4f max=%.4f mean(center)=%.4f ===%n",
				mm.minVal, mm.maxVal, meanOfRegion(alpha, 0.45, 0.35, 0.2, 0.3));
			assertThat(mm.minVal).isGreaterThanOrEqualTo(0d);
			assertThat(mm.maxVal).isLessThanOrEqualTo(1.0001d);
			// 主体（大致居中）应显著高于四角背景
			double center = meanOfRegion(alpha, 0.4, 0.3, 0.25, 0.4);
			double corner = meanOfRegion(alpha, 0.0, 0.0, 0.1, 0.1);
			assertThat(center).as("主体中心 alpha 应显著高于左上角背景").isGreaterThan(corner + 0.3);
		}
	}

	/**
	 * min-max 拉伸对「低动态范围」输入必须真的起作用：flat.png（近纯色）原始输出极差仅 ~0.002。
	 *
	 * <p>⚠️ 注意断言位置：min-max 只保证 <b>320×320 的模型输出</b>被铺满 [0,1]。
	 * 掩码随后要 resize 回原尺寸，而 INTER_LINEAR 会对邻域做加权平均，
	 * 把孤立的单像素极大值稀释掉（实测 320×320 的 1.0 → 240×180 的 ~0.57~0.67）。
	 * 因此**不能在最终掩码上断言 max == 1**，只能断言「拉伸显著抬高了动态范围」。
	 */
	@Test
	void minMaxNormalizeShouldStretchLowDynamicRangeInput() throws Exception {
		byte[] bytes = readResource("/flat.png");
		double onMax;
		try (MattingResult on = engine.matteBytes(bytes)) {
			Core.MinMaxLocResult mm = Core.minMaxLoc(on.getAlpha());
			onMax = mm.maxVal;
			System.out.printf("=== flat.png minMaxNormalize=true: min=%.6f max=%.6f ===%n",
				mm.minVal, mm.maxVal);
		}
		MattingConfig raw = MattingConfig.builder()
			.modelPath(modelPath.toString())
			.minMaxNormalize(false)
			.build();
		double offMax;
		try (MattingEngine rawEngine = MattingEngine.create(raw);
			 MattingResult off = rawEngine.matteBytes(bytes)) {
			Core.MinMaxLocResult mm = Core.minMaxLoc(off.getAlpha());
			offMax = mm.maxVal;
			System.out.printf("=== flat.png minMaxNormalize=false: min=%.8f max=%.8f ===%n",
				mm.minVal, mm.maxVal);
			assertThat(mm.maxVal).as("关闭归一化后应保留模型原始小值").isLessThan(0.01d);
		}
		assertThat(onMax).as("开启 min-max 后动态范围应被显著抬高").isGreaterThan(0.3d);
		assertThat(onMax / Math.max(offMax, 1e-9))
			.as("拉伸放大约两个数量级").isGreaterThan(50d);
	}

	// ---------------------------------------------------------------- 输出形态

	@Test
	void cutoutBytesShouldProduceBgraPng() throws Exception {
		byte[] bytes = readResource("/subject.png");
		byte[] png = engine.cutoutBytes(bytes);
		assertThat(png).isNotEmpty();
		Mat decoded = decode(png);
		try {
			assertThat(decoded.channels()).as("透明底输出应为 4 通道 BGRA").isEqualTo(4);
			assertThat(decoded.cols()).isEqualTo(480);
			assertThat(decoded.rows()).isEqualTo(360);
		} finally {
			decoded.release();
		}
	}

	@Test
	void cutoutOnColorBytesShouldCompositeOnRequestedBackground() throws Exception {
		byte[] bytes = readResource("/subject.png");
		byte[] png = engine.cutoutOnColorBytes(bytes, new int[]{0, 128, 255});
		Mat decoded = decode(png);
		try {
			assertThat(decoded.channels()).isEqualTo(3);
			// 左上角应是（被 alpha≈0 压平的）蓝色底 —— BGR 顺序下为 (255, 128, 0)
			double[] corner = decoded.get(2, 2);
			System.out.printf("=== corner BGR = [%.0f, %.0f, %.0f] ===%n",
				corner[0], corner[1], corner[2]);
			assertThat(corner[2]).as("红色分量应接近 0").isLessThan(30d);
			assertThat(corner[1]).as("绿色分量应接近 128").isBetween(100d, 156d);
			assertThat(corner[0]).as("蓝色分量应接近 255").isGreaterThan(225d);
		} finally {
			decoded.release();
		}
	}

	@Test
	void matteBinaryBytesShouldProduceSingleChannelMask() throws Exception {
		byte[] bytes = readResource("/subject.png");
		byte[] png = engine.matteBinaryBytes(bytes);
		Mat decoded = decode(png);
		try {
			assertThat(decoded.channels()).as("二值掩码应为单通道").isEqualTo(1);
			assertThat(decoded.cols()).isEqualTo(480);
			assertThat(decoded.rows()).isEqualTo(360);
			Mat gray = new Mat();
			try {
				Core.extractChannel(decoded, gray, 0);
				Core.MinMaxLocResult mm = Core.minMaxLoc(gray);
				assertThat(mm.minVal).as("二值掩码只允许 0").isEqualTo(0d);
				assertThat(mm.maxVal).as("二值掩码只允许 255").isEqualTo(255d);
			} finally {
				gray.release();
			}
		} finally {
			decoded.release();
		}
	}

	@Test
	void pathInputShouldWorkAndMissingFileShouldThrow() throws Exception {
		byte[] bytes = readResource("/subject.png");
		Path tmp = Files.createTempFile("mica-ai-matting-", ".png");
		try {
			Files.write(tmp, bytes);
			byte[] png = engine.cutoutPath(tmp.toString());
			assertThat(png).isNotEmpty();
		} finally {
			Files.deleteIfExists(tmp);
		}
		assertThatThrownBy(() -> engine.cutoutPath("no-such-file-xyz.png"))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("文件不存在");
	}

	@Test
	void nullOrEmptyInputShouldBeHandledGracefully() {
		assertThat(engine.alpha((Mat) null)).isNull();
		assertThat(engine.matte(new Mat())).isNull();
	}

	@Test
	void binaryThresholdShouldAffectForegroundRatio() throws Exception {
		byte[] bytes = readResource("/subject.png");
		MattingConfig relaxed = MattingConfig.builder()
			.modelPath(modelPath.toString())
			.binaryThreshold(0.1f)
			.build();
		try (MattingEngine relaxedEngine = MattingEngine.create(relaxed)) {
			assertThat(countForeground(relaxedEngine.matteBinaryBytes(bytes), 480, 360))
				.as("低阈值应保留更多前景")
				.isGreaterThan(countForeground(engine.matteBinaryBytes(bytes), 480, 360));
		}
	}

	// ---------------------------------------------------------------- 可插拔模型契约

	/**
	 * 输出选择策略必须真的生效，而不是「配了也不用」。
	 *
	 * <p>rembg 导出的输出名是数字（1959..1965），走 {@code AUTO} 会命中「取首个」兜底分支；
	 * {@code D0} 要求精确名为 {@code d0}，对这种导出必须快速失败——这层差异证明策略被读取了。
	 */
	@Test
	void outputSelectStrategyShouldActuallyBeApplied() {
		MattingConfig auto = MattingConfig.builder()
			.modelPath(modelPath.toString())
			.outputSelect(MattingOutputSelect.AUTO)
			.build();
		try (MattingEngine probe = MattingEngine.create(auto)) {
			assertThat(probe.getConfig().getOutputSelect()).isEqualTo(MattingOutputSelect.AUTO);
		}

		MattingConfig d0Only = MattingConfig.builder()
			.modelPath(modelPath.toString())
			.outputSelect(MattingOutputSelect.D0)
			.build();
		assertThatThrownBy(() -> MattingEngine.create(d0Only))
			.as("D0 策略在数字命名的 rembg 导出上必须快速失败")
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("d0");
	}

	/**
	 * {@code FIRST} 策略（不依赖任何节点名）在 rembg 导出上应与 {@code AUTO} 等价。
	 */
	@Test
	void firstStrategyShouldMatchAutoOnRembgExport() throws Exception {
		byte[] bytes = readResource("/subject.png");
		MattingConfig firstConfig = MattingConfig.builder()
			.modelPath(modelPath.toString())
			.outputSelect(MattingOutputSelect.FIRST)
			.build();
		try (MattingEngine firstEngine = MattingEngine.create(firstConfig)) {
			try (MattingResult a = engine.matteBytes(bytes);
				 MattingResult b = firstEngine.matteBytes(bytes)) {
				double mae = meanAbsDiff(a.getAlpha(), b.getAlpha());
				System.out.printf("=== AUTO vs FIRST 掩码 MAE=%.8f ===%n", mae);
				assertThat(mae).as("同一模型上 AUTO 与 FIRST 应取到同一个输出")
					.isLessThan(1e-6);
			}
		}
	}

	/**
	 * 外置模型接入：指向 {@code model-path} 的任意 U²-Net 族模型都应能加载。
	 *
	 * <p>这里只验证「超限外置模型可加载 + 契约自洽」，不集成 168MB 权重——
	 * 按 AGENTS.md §6.2 该类模型不入库，缺失时跳过。
	 */
	@Test
	void externalModelShouldBeLoadableWhenProvided() {
		String external = System.getProperty("mica.ai.matting.externalModel");
		Assumptions.assumeTrue(external != null && Files.exists(Paths.get(external)),
			"未通过 -Dmica.ai.matting.externalModel 提供外置模型，跳过");
		MattingConfig config = MattingConfig.builder()
			.modelPath(external)
			.build();
		try (MattingEngine externalEngine = MattingEngine.create(config)) {
			assertThat(externalEngine.modelInputSize()).isEqualTo(320);
			assertThat(externalEngine.outputCount()).isEqualTo(7);
		}
	}

	// ---------------------------------------------------------------- 辅助

	private static double meanAbsDiff(Mat a, Mat b) {
		assertThat(a.size()).isEqualTo(b.size());
		Mat diff = new Mat();
		try {
			Core.absdiff(a, b, diff);
			return Core.mean(diff).val[0];
		} finally {
			diff.release();
		}
	}

	private static int countForeground(byte[] png, int width, int height) {
		Mat decoded = decode(png);
		Mat gray = new Mat();
		try {
			Core.extractChannel(decoded, gray, 0);
			int[] count = new int[]{0};
			byte[] row = new byte[width];
			for (int y = 0; y < height; y++) {
				gray.get(y, 0, row);
				for (byte b : row) {
					if ((b & 0xFF) > 127) {
						count[0]++;
					}
				}
			}
			return count[0];
		} finally {
			gray.release();
			decoded.release();
		}
	}

	private static double meanOfRegion(Mat alpha, double xRatio, double yRatio,
									   double wRatio, double hRatio) {
		int x = (int) (alpha.cols() * xRatio);
		int y = (int) (alpha.rows() * yRatio);
		int w = Math.max(1, (int) (alpha.cols() * wRatio));
		int h = Math.max(1, (int) (alpha.rows() * hRatio));
		x = Math.min(x, alpha.cols() - 1);
		y = Math.min(y, alpha.rows() - 1);
		w = Math.min(w, alpha.cols() - x);
		h = Math.min(h, alpha.rows() - y);
		Mat region = alpha.submat(y, y + h, x, x + w);
		try {
			return Core.mean(region).val[0];
		} finally {
			region.release();
		}
	}

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

	private static int[] imageSize(byte[] bytes) {
		Mat mat = decode(bytes);
		try {
			return new int[]{mat.cols(), mat.rows()};
		} finally {
			mat.release();
		}
	}

	private static byte[] readResource(String name) throws Exception {
		try (InputStream in = MattingIntegrationTest.class.getResourceAsStream(name)) {
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
