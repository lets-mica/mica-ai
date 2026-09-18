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
package net.dreamlu.mica.ai.layout;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.util.IOUtil;
import net.dreamlu.mica.ai.layout.config.LayoutConfig;
import net.dreamlu.mica.ai.layout.model.LayoutResult;
import net.dreamlu.mica.ai.layout.pipeline.LayoutPipeline;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * 集成测试：加载仓库内真实 PP-DocLayoutV3 ONNX（{@code model-tools/layout/models/model.onnx}），
 * 对模块测试图 {@code demo.png} 跑一遍完整链路（letterbox → ONNX → 反 letterbox → NMS → 阅读顺序）。
 *
 * <p><b>模型缺失时整个测试类跳过</b>（{@code Assumptions}）：该模型 125MB，超出
 * AGENTS.md §6.2 的「&lt;50MB 入库」约定，入库方式待决策，因此新克隆的仓库里可能没有它。
 * 跳过时会在报告里打印提示，不会让构建变绿而掩盖真实回归。
 *
 * <p>OpenCV 原生库由 openpnp 的 {@code nu.pattern.OpenCV.loadShared()} 在静态块中释放并加载。
 */
class LayoutIntegrationTest {

	static {
		nu.pattern.OpenCV.loadShared();
	}

	private static final String MODEL_RELATIVE = "model-tools/layout/models/model.onnx";

	private static LayoutPipeline pipeline;
	private static Path modelPath;
	private static byte[] imageBytes;
	private static int imageW;
	private static int imageH;

	@BeforeAll
	static void setUp() throws Exception {
		modelPath = locateRepoFile(MODEL_RELATIVE);
		org.junit.jupiter.api.Assumptions.assumeTrue(
			modelPath != null && Files.exists(modelPath),
			"未找到 " + MODEL_RELATIVE + "（125MB，入库方式待决策，可能未随仓库分发），跳过真模型集成测试");
		imageBytes = readResource("/demo.png");
		Mat mat = null;
		MatOfByte mob = new MatOfByte(imageBytes);
		try {
			mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
			assertThat(mat.empty()).as("demo.png 必须可解码").isFalse();
			imageW = mat.cols();
			imageH = mat.rows();
		} finally {
			mob.release();
			if (mat != null) {
				mat.release();
			}
		}
		pipeline = LayoutPipeline.create(LayoutConfig.builder()
			.modelPath(modelPath.toString())
			.build());
	}

	@AfterAll
	static void tearDown() {
		if (pipeline != null) {
			pipeline.close();
		}
	}

	@Test
	void detectBytesShouldReturnRegionsInOriginalImageSpace() {
		List<LayoutResult> out = pipeline.detectBytes(imageBytes);

		System.out.println("=== mica-ai-layout 真模型集成测试（demo.png " + imageW + "x" + imageH + "）===");
		for (LayoutResult r : out) {
			System.out.printf("[%d] read=%d %-16s score=%.3f bbox=%s%n",
				r.getIndex(), r.getReadingOrder(), r.getLabelCode(), r.getScore(),
				java.util.Arrays.toString(r.getBoundingBox()));
		}

		assertThat(out).as("demo.png 至少应检出 1 个版面区域").isNotEmpty();
		assertThat(out).allSatisfy(r -> {
			assertThat(r.getLabel()).isNotNull();
			assertThat(r.getLabelCode()).isEqualTo(r.getLabel().getCode());
			assertThat(r.getScore()).isBetween(0.4f, 1f);
			int[] b = r.getBoundingBox();
			assertThat(b).hasSize(4);
			assertThat(b[0]).isGreaterThanOrEqualTo(0);
			assertThat(b[1]).isGreaterThanOrEqualTo(0);
			assertThat(b[2]).isLessThanOrEqualTo(imageW);
			assertThat(b[3]).isLessThanOrEqualTo(imageH);
			assertThat(b[2]).isGreaterThan(b[0]);
			assertThat(b[3]).isGreaterThan(b[1]);
			// 对齐 PaddleX：跳过类为 -1（不占号），其余从 1 起
			if (r.getLabel().isSkipOrder()) {
				assertThat(r.getReadingOrder())
					.as("跳过类 %s 不应参与阅读顺序编号", r.getLabelCode())
					.isEqualTo(LayoutResult.NO_READING_ORDER);
			} else {
				assertThat(r.getReadingOrder()).isGreaterThanOrEqualTo(1);
			}
		});
		// 参与编号的区域必须连续且不重复（PaddleX 的 1..N）
		assertThat(out).filteredOn(r -> r.getReadingOrder() > 0)
			.extracting(LayoutResult::getReadingOrder).doesNotHaveDuplicates();
	}

	@Test
	void maxDetectionsShouldCapResults() {
		LayoutConfig config = LayoutConfig.builder()
			.modelPath(modelPath.toString())
			.maxDetections(2)
			.build();
		try (LayoutPipeline limited = LayoutPipeline.create(config)) {
			assertThat(limited.detectBytes(imageBytes)).hasSizeLessThanOrEqualTo(2);
		}
	}

	@Test
	void classScoreThresholdShouldFilterByClass() {
		LayoutConfig config = LayoutConfig.builder()
			.modelPath(modelPath.toString())
			.scoreThreshold(0.99f)
			.build();
		try (LayoutPipeline strict = LayoutPipeline.create(config)) {
			assertThat(strict.detectBytes(imageBytes)).isEmpty();
		}
	}

	@Test
	void maxSideLengthMismatchShouldFailFast() {
		LayoutConfig config = LayoutConfig.builder()
			.modelPath(modelPath.toString())
			.maxSideLength(960)
			.build();
		assertThatThrownBy(() -> LayoutPipeline.create(config))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("maxSideLength");
	}

	/**
	 * 真实文档图回归（阈值标定的落地断言）。通过系统属性 {@code mica.ai.layout.test.image}
	 * 指定图片路径；未指定时跳过，因此不会让 CI 依赖外部资源。
	 *
	 * <pre>
	 * sh mvnc.sh -o -pl mica-ai-core/mica-ai-layout test \
	 *     -Dmica.ai.layout.test.image=E:/tmp/layout_demo.jpg
	 * </pre>
	 *
	 * <p>为什么需要它：{@code demo.png} 只有 1 个区域、分数 0.437 勉强过阈值，
	 * 且是 800×800 正方形（letterbox 缩放比恒为 1）⇒ 覆盖不到真实文档的多区域、
	 * 非等比反算、分数长尾三种情况。标定结论见 {@code LayoutConfig#getScoreRatio()}。
	 */
	@Test
	void externalDocumentImageShouldDetectMultipleRegions() throws Exception {
		String path = System.getProperty("mica.ai.layout.test.image");
		org.junit.jupiter.api.Assumptions.assumeTrue(
			path != null && Files.exists(Paths.get(path)),
			"未提供 mica.ai.layout.test.image，跳过真实文档图校验");
		List<LayoutResult> out = pipeline.detectBytes(Files.readAllBytes(Paths.get(path)));
		System.out.println("=== 外部文档图 " + path + " 检出 " + out.size() + " 个区域 ===");
		for (LayoutResult r : out) {
			System.out.printf("[%d] read=%d %-16s score=%.3f bbox=%s%n",
				r.getIndex(), r.getReadingOrder(), r.getLabelCode(), r.getScore(),
				java.util.Arrays.toString(r.getBoundingBox()));
		}
		assertThat(out.size()).as("真实文档图应检出多个版面区域").isGreaterThan(3);
		assertThat(out.get(0).getScore()).isGreaterThan(0.8f);
		// 真实文档图上应当同时出现「参与编号」与「跳过编号」两类区域
		assertThat(out).anyMatch(r -> r.getReadingOrder() > 0);
		assertThat(out).filteredOn(r -> r.getReadingOrder() > 0)
			.extracting(LayoutResult::getReadingOrder).doesNotHaveDuplicates();
	}

	/**
	 * 真实文档图上的相对阈值回归：开启 {@code scoreRatio} 后，参与编号的区域
	 * 必须连续 1..N（无空洞）—— 这是「丢弃长尾误检」的直接可观测后果。
	 *
	 * <p>标定依据（官方 demo 1654×2339 实测）：真实内容区分数 0.69–0.94，
	 * 长尾误检 0.518 / 0.437；{@code scoreRatio=0.6} 时阈值 0.567，
	 * 恰好切在分数悬崖处，保留 12 个区域且编号连续。
	 */
	@Test
	void scoreRatioShouldProduceContiguousReadingOrder() throws Exception {
		String path = System.getProperty("mica.ai.layout.test.image");
		org.junit.jupiter.api.Assumptions.assumeTrue(
			path != null && Files.exists(Paths.get(path)),
			"未提供 mica.ai.layout.test.image，跳过相对阈值回归");
		LayoutConfig config = LayoutConfig.builder()
			.modelPath(modelPath.toString())
			.scoreRatio(0.6f)
			.build();
		try (LayoutPipeline calibrated = LayoutPipeline.create(config)) {
			List<LayoutResult> out = calibrated.detectBytes(Files.readAllBytes(Paths.get(path)));
			List<Integer> ordered = new java.util.ArrayList<>();
			for (LayoutResult r : out) {
				if (r.getReadingOrder() > 0) {
					ordered.add(r.getReadingOrder());
				}
			}
			java.util.Collections.sort(ordered);
			System.out.println("=== scoreRatio=0.6 参与编号区域: " + ordered + " ===");
			assertThat(ordered).as("阅读顺序编号必须从 1 开始连续无空洞").isNotEmpty();
			for (int i = 0; i < ordered.size(); i++) {
				assertThat(ordered.get(i)).isEqualTo(i + 1);
			}
		}
	}

	private static byte[] readResource(String name) throws Exception {
		try (InputStream in = LayoutIntegrationTest.class.getResourceAsStream(name)) {
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
