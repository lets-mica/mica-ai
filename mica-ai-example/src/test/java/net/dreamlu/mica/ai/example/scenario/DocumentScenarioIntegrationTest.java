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
package net.dreamlu.mica.ai.example.scenario;

import net.dreamlu.mica.ai.classify.config.ClassifyConfig;
import net.dreamlu.mica.ai.classify.core.ClassifyEngine;
import net.dreamlu.mica.ai.classify.model.ClassifyResult;
import net.dreamlu.mica.ai.docalign.config.DocAlignConfig;
import net.dreamlu.mica.ai.docalign.core.DocAlignEngine;
import net.dreamlu.mica.ai.enhance.config.EnhanceConfig;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.example.config.ClassifyModelProperties;
import net.dreamlu.mica.ai.example.config.ClassifyModelRegistry;
import net.dreamlu.mica.ai.example.pipeline.document.DocumentOptions;
import net.dreamlu.mica.ai.example.pipeline.document.DocumentPipeline;
import net.dreamlu.mica.ai.example.pipeline.document.DocumentResult;
import net.dreamlu.mica.ai.textline.config.TextLineConfig;
import net.dreamlu.mica.ai.textline.core.TextLineEngine;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * 智能文档流水线集成测试：加载真实 ONNX 模型跑完整链路。
 *
 * <p>模型缺失时整类跳过，不阻塞新克隆仓库的构建。
 */
class DocumentScenarioIntegrationTest {

	private static DocAlignEngine docAlignEngine;
	private static ClassifyEngine orientationEngine;
	private static ClassifyEngine clarityEngine;
	private static TextLineEngine textLineEngine;
	private static EnhanceEngine enhanceEngine;
	private static DocumentPipeline pipeline;

	@BeforeAll
	static void setUp() {
		nu.pattern.OpenCV.loadLocally();

		Path docalignModel = repoFile(
			"model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx");
		Path orientationModel = repoFile(
			"model-tools/classify/models/text_image_orientation.onnx");
		Path orientationLabels = repoFile(
			"model-tools/classify/models/text_image_orientation_labels.txt");
		Path clarityModel = repoFile("model-tools/classify/models/clarity_assessment.onnx");
		Path clarityLabels = repoFile(
			"model-tools/classify/models/clarity_assessment_labels.txt");
		Path textlineModel = repoFile(
			"model-tools/textline/models/PP-LCNet_x1_0_textline_ori.onnx");
		Path enhanceModel = repoFile("model-tools/enhance/models/realesr-general-x4v3.onnx");
		if (docalignModel == null || orientationModel == null || orientationLabels == null
			|| clarityModel == null || clarityLabels == null || textlineModel == null
			|| enhanceModel == null) {
			return;
		}

		docAlignEngine = DocAlignEngine.create(DocAlignConfig.builder()
			.modelPath(docalignModel.toAbsolutePath().toString()).build());
		orientationEngine = ClassifyEngine.create(ClassifyConfig.builder()
			.modelVersion("text_image_orientation")
			.modelPath(orientationModel.toAbsolutePath().toString())
			.labelsPath(orientationLabels.toAbsolutePath().toString())
			.resizeShort(256).multiLabel(false).topK(1).build());
		clarityEngine = ClassifyEngine.create(ClassifyConfig.builder()
			.modelVersion("clarity_assessment")
			.modelPath(clarityModel.toAbsolutePath().toString())
			.labelsPath(clarityLabels.toAbsolutePath().toString())
			.resizeShort(256).multiLabel(false).topK(1).build());
		textLineEngine = TextLineEngine.create(TextLineConfig.builder()
			.modelPath(textlineModel.toAbsolutePath().toString()).build());
		enhanceEngine = EnhanceEngine.create(EnhanceConfig.builder()
			.modelPath(enhanceModel.toAbsolutePath().toString()).build());

		ClassifyModelProperties properties = new ClassifyModelProperties();
		Map<String, ClassifyModelProperties.ModelEntry> models = new LinkedHashMap<>();
		ClassifyModelProperties.ModelEntry clarity = new ClassifyModelProperties.ModelEntry();
		clarity.setModelPath(clarityModel.toAbsolutePath().toString());
		clarity.setLabelsPath(clarityLabels.toAbsolutePath().toString());
		models.put("clarity", clarity);
		properties.setModels(models);

		pipeline = new DocumentPipeline(docAlignEngine, orientationEngine,
			new ClassifyModelRegistry(properties), enhanceEngine, null, textLineEngine);
	}

	@Test
	@DisplayName("整图方向：标签是当前顺时针角度，修正后应回到 0°（回转验证）")
	void orientationStageShouldStraightenRotatedImage() {
		assumeReady();
		DocumentOptions options = DocumentOptions.builder()
			.align(false).orientation(true).enhance(false).layout(false).textLine(false)
			.build();

		// rot90.jpg = 正立文档顺时针转 90°
		DocumentResult result = pipeline.process(
			loadFixture("text-orientation/rot90.jpg"), options);
		assertThat(result.getOrientation()).isNotNull();
		assertThat(result.getOrientation().getLabel()).isEqualTo("90");
		assertThat(result.getOrientation().isApplied()).isTrue();
		// 关键断言：修正方向必须正确——处理后的图再分类应回到 0°
		// 若按「顺时针转 label 度」实现，这里会得到 180
		ClassifyResult after = orientationEngine.classifyBytes(result.getOutputPng());
		assertThat(after.topLabel()).isEqualTo("0");

		// rot270.jpg = 正立文档顺时针转 270°
		DocumentResult result270 = pipeline.process(
			loadFixture("text-orientation/rot270.jpg"), options);
		assertThat(result270.getOrientation().getLabel()).isEqualTo("270");
		assertThat(orientationEngine.classifyBytes(result270.getOutputPng()).topLabel())
			.isEqualTo("0");
	}

	@Test
	@DisplayName("整图方向：正立文档不应被旋转")
	void orientationStageShouldKeepUprightDocument() {
		assumeReady();
		DocumentOptions options = DocumentOptions.builder()
			.align(false).orientation(true).enhance(false).layout(false).textLine(false)
			.build();
		DocumentResult result = pipeline.process(
			loadFixture("text-orientation/rot0.jpg"), options);
		assertThat(result.getOrientation().getLabel()).isEqualTo("0");
		assertThat(result.getOrientation().isApplied()).isFalse();
	}

	@Test
	@DisplayName("拉正阶段：真实斜拍文档应检出四角并透视拉平")
	void alignStageShouldRectifySkewedDocument() {
		assumeReady();
		DocumentOptions options = DocumentOptions.builder()
			.align(true).orientation(false).enhance(false).layout(false).textLine(false)
			.build();
		DocumentResult result = pipeline.process(loadFixture("doc_warp.png"), options);

		assertThat(result.getAlign()).isNotNull();
		assertThat(result.getAlign().getCorners()).hasSize(4);
		assertThat(result.getAlign().getMinCornerScore()).isGreaterThan(0d);
		assertThat(result.getAlign().isApplied()).isTrue();
		assertThat(result.getOutputPng()).isNotEmpty();
		assertThat(result.getWidth()).isGreaterThan(0);
		assertThat(result.getHeight()).isGreaterThan(0);
	}

	@Test
	@DisplayName("清晰度阶段：清晰文档判为 clear 且不触发超分")
	void clarityStageShouldNotUpscaleSharpDocument() {
		assumeReady();
		DocumentOptions options = DocumentOptions.builder()
			.align(false).orientation(false).enhance(true).layout(false).textLine(false)
			.build();
		DocumentResult result = pipeline.process(
			loadFixture("text-orientation/rot0.jpg"), options);
		assertThat(result.getClarity()).isNotNull();
		assertThat(result.getClarity().getLabel()).isEqualTo("clear");
		assertThat(result.getClarity().isApplied()).isFalse();
	}

	@Test
	@DisplayName("阶段报告：五个阶段都要有记录，layout 未启用时降级说明")
	void stagesShouldAlwaysBeReported() {
		assumeReady();
		DocumentResult result = pipeline.process(
			loadFixture("doc_warp.png"),
			DocumentOptions.builder().build());

		assertThat(result.getStages()).hasSize(5);
		assertThat(result.getStages().get(0).getName()).isEqualTo("align");
		assertThat(result.getStages().get(1).getName()).isEqualTo("orientation");
		assertThat(result.getStages().get(2).getName()).isEqualTo("layout");
		assertThat(result.getStages().get(3).getName()).isEqualTo("textLine");
		assertThat(result.getStages().get(4).getName()).isEqualTo("clarity");
		// layout 模型未装配（125MB 不入库），阶段应降级而不是抛异常
		assertThat(result.getStages().get(2).isApplied()).isFalse();
		assertThat(result.getStages().get(2).getDetail()).contains("layout");
		assertThat(result.getBlocks()).isEmpty();
	}

	private static void assumeReady() {
		assumeTrue(pipeline != null, "相关模型缺失，跳过智能文档流水线集成测试");
	}

	/**
	 * 读取仓库内其它模块的测试素材（相对仓库根）。
	 */
	private static byte[] loadFixture(String relative) {
		Path path = repoFile("mica-ai-core/mica-ai-classify/src/test/resources/" + relative);
		if (path == null) {
			path = repoFile("mica-ai-core/mica-ai-docalign/src/test/resources/" + relative);
		}
		assumeTrue(path != null, "测试素材缺失: " + relative);
		try {
			return Files.readAllBytes(path);
		} catch (IOException e) {
			throw new IllegalStateException("读取测试素材失败: " + relative, e);
		}
	}

	/**
	 * 从当前工作目录向上查找仓库根。
	 */
	private static Path repoFile(String relative) {
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

}
