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
package net.dreamlu.mica.ai.face.quality;

import net.dreamlu.mica.ai.face.config.ExpressionConfig;
import net.dreamlu.mica.ai.face.config.FaceQualityConfig;
import net.dreamlu.mica.ai.face.expression.ExpressionRecognizer;
import net.dreamlu.mica.ai.face.expression.ExpressionResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * mica-ai-face 真模型集成测试：加载仓库内 eDifFIQA 与表情识别 ONNX
 * （{@code model-tools/face/models/}），跑通「模型加载 → 推理 → 结果取值域」完整链路。
 *
 * <p>这两个模型入库时<b>没有</b>任何集成测试兜底，导致一次被截断的下载
 * （ONNX protobuf 声明的 graph 长度大于文件实际长度）长期潜伏，
 * 直到被 Starter 按需装配时才以「Protobuf parsing failed」暴露。
 * 本测试类就是那道防线：模型一损坏，装配阶段即失败。
 *
 * <p>OpenCV 原生库由 {@code nu.pattern.OpenCV.loadLocally()} 在静态块中加载，
 * 避开 Surefire forkCount=0 下 native 库多次加载的路径问题。
 */
@EnabledIf("isModelPresent")
class FaceQualityExpressionIntegrationTest {

	static {
		nu.pattern.OpenCV.loadLocally();
	}

	private static final Path QUALITY_PATH =
		repoPath("model-tools/face/models/ediffiqa_tiny_jun2024.onnx");
	private static final Path EXPRESSION_PATH =
		repoPath("model-tools/face/models/facial_expression_recognition_mobilefacenet_2022july.onnx");

	static boolean isModelPresent() {
		return Files.exists(QUALITY_PATH) && Files.exists(EXPRESSION_PATH);
	}

	private static Path repoPath(String relativePath) {
		Path dir = Paths.get("").toAbsolutePath();
		while (dir != null && !Files.exists(dir.resolve("model-tools"))) {
			dir = dir.getParent();
		}
		return dir == null ? Paths.get(relativePath) : dir.resolve(relativePath);
	}

	/**
	 * 构造一张 112×112 的「已对齐」合成脸：浅肤色底 + 眼 + 嘴。
	 * 不追求贴近训练分布，只保证是合法的 BGR 输入。
	 */
	private static Mat buildAlignedFaceStub() {
		Mat img = new Mat(112, 112, CvType.CV_8UC3, new Scalar(200, 180, 160));
		Imgproc.circle(img, new org.opencv.core.Point(45, 45), 6, new Scalar(20, 20, 20), -1);
		Imgproc.circle(img, new org.opencv.core.Point(67, 45), 6, new Scalar(20, 20, 20), -1);
		Imgproc.ellipse(img, new org.opencv.core.Point(56, 75), new org.opencv.core.Size(15, 7),
			0, 0, 180, new Scalar(40, 40, 40), 2);
		return img;
	}

	@Test
	void qualityAssessorLoadsRealModelAndScoresInUnitRange() {
		Mat aligned = buildAlignedFaceStub();
		try (FaceQualityAssessor assessor = FaceQualityAssessor.create(FaceQualityConfig.builder()
			.modelPath(QUALITY_PATH.toString())
			.threshold(0.5f)
			.build())) {
			FaceQualityResult result = assessor.assess(aligned);
			assertThat(result).isNotNull();
			assertThat(result.getScore()).isBetween(0f, 1f);
			// passed 与阈值判定一致
			assertThat(result.passed(0.5f)).isEqualTo(result.getScore() >= 0.5f);
		} finally {
			aligned.release();
		}
	}

	@Test
	void expressionRecognizerLoadsRealModelAndReturnsSevenClassProbabilities() {
		Mat aligned = buildAlignedFaceStub();
		try (ExpressionRecognizer recognizer = ExpressionRecognizer.create(ExpressionConfig.builder()
			.modelPath(EXPRESSION_PATH.toString())
			.build())) {
			ExpressionResult result = recognizer.recognize(aligned);
			assertThat(result).isNotNull();
			assertThat(result.getExpression()).isNotNull();
			assertThat(result.getScore()).isBetween(0f, 1f);
			float[] probs = result.getProbabilities();
			assertThat(probs).hasSize(7);
			double sum = 0d;
			for (float p : probs) {
				assertThat(p).isBetween(0f, 1f);
				sum += p;
			}
			// softmax 归一化 → 概率和为 1
			assertThat(sum).isCloseTo(1.0d, org.assertj.core.data.Offset.offset(1e-3d));
		} finally {
			aligned.release();
		}
	}
}
