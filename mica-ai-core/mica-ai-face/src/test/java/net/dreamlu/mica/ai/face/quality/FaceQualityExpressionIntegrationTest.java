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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
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

import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

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

	/**
	 * 构造一张<b>空间与通道都不均匀</b>的彩色脸。
	 *
	 * <p>布局回归测试必须用这种输入：若整幅图三通道取值相同，
	 * 通道反序与 HWC/CHW 错配都可能碰巧给出相同结果，测不出问题。
	 */
	private static Mat buildColorfulAlignedFace() {
		Mat img = new Mat(112, 112, CvType.CV_8UC3, new Scalar(90, 140, 200));
		Imgproc.rectangle(img, new org.opencv.core.Point(8, 8), new org.opencv.core.Point(52, 52),
			new Scalar(30, 200, 70), -1);
		Imgproc.circle(img, new org.opencv.core.Point(70, 40), 14, new Scalar(240, 60, 20), -1);
		Imgproc.ellipse(img, new org.opencv.core.Point(56, 78), new org.opencv.core.Size(20, 9),
			0, 0, 180, new Scalar(10, 20, 250), 3);
		Imgproc.rectangle(img, new org.opencv.core.Point(0, 92), new org.opencv.core.Point(111, 111),
			new Scalar(120, 130, 140), -1);
		return img;
	}

	/**
	 * 官方管线参考实现（<b>独立于生产代码</b>）。
	 *
	 * <p>对应 OpenCV Zoo {@code facial_fer_model.py} / {@code ediffiqa.py}：
	 * BGR→RGB、{@code /255}、{@code (x - mean) / std}、通道前置成 CHW。
	 * 写法（平面外层循环）与生产代码（像素外层循环）刻意不同，
	 * 避免「同一种误解自证正确」。
	 *
	 * @param bgr  BGR 输入
	 * @param mean 三通道均值（RGB 顺序）
	 * @param std  三通道标准差（RGB 顺序）
	 * @return CHW 布局的归一化 float 数组
	 */
	private static float[] officialChw(Mat bgr, float[] mean, float[] std) {
		int h = bgr.rows();
		int w = bgr.cols();
		int hw = h * w;
		byte[] raw = new byte[hw * 3];
		bgr.get(0, 0, raw);
		float[] chw = new float[3 * hw];
		for (int c = 0; c < 3; c++) {
			// BGR 输入里 R 是第 2 通道；输出 CHW 的平面顺序为 R、G、B
			int srcChannel = 2 - c;
			for (int i = 0; i < hw; i++) {
				int v = raw[i * 3 + srcChannel] & 0xFF;
				chw[c * hw + i] = ((v / 255f) - mean[c]) / std[c];
			}
		}
		return chw;
	}

	private static float[] runExpressionOnnx(float[] chw) throws OrtException {
		OrtEnvironment env = OrtEnvironment.getEnvironment();
		try (OrtSession session = env.createSession(EXPRESSION_PATH.toString());
			 OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw),
				 new long[]{1, 3, 112, 112})) {
			Map<String, OnnxTensor> inputs = new HashMap<>();
			inputs.put(session.getInputNames().iterator().next(), tensor);
			try (OrtSession.Result result = session.run(inputs)) {
				float[][] values = (float[][]) result.get(0).getValue();
				return values[0];
			}
		}
	}

	private static float runQualityOnnx(float[] chw) throws OrtException {
		OrtEnvironment env = OrtEnvironment.getEnvironment();
		try (OrtSession session = env.createSession(QUALITY_PATH.toString());
			 OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw),
				 new long[]{1, 3, 112, 112})) {
			Map<String, OnnxTensor> inputs = new HashMap<>();
			inputs.put(session.getInputNames().iterator().next(), tensor);
			try (OrtSession.Result result = session.run(inputs)) {
				float[][] values = (float[][]) result.get(0).getValue();
				return values[0][0];
			}
		}
	}

	private static float[] softmax(float[] logits) {
		float max = logits[0];
		for (float v : logits) {
			max = Math.max(max, v);
		}
		float[] exp = new float[logits.length];
		float sum = 0f;
		for (int i = 0; i < logits.length; i++) {
			exp[i] = (float) Math.exp(logits[i] - max);
			sum += exp[i];
		}
		for (int i = 0; i < logits.length; i++) {
			exp[i] /= sum;
		}
		return exp;
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

	/**
	 * 表情识别的生产预处理必须与官方 OpenCV Zoo 管线<b>逐位一致</b>。
	 *
	 * <p>历史事故：生产侧按 HWC 填充内存却声明成 NCHW，模型收到被打乱的图像，
	 * 输出退化成恒定的「平静」。本测试用官方管线的独立实现做对照，
	 * 任何布局 / 通道序 / 归一化的偏差都会立刻暴露。
	 */
	@Test
	void expressionPreprocessMatchesOfficialPipeline() throws OrtException {
		Mat aligned = buildColorfulAlignedFace();
		try (ExpressionRecognizer recognizer = ExpressionRecognizer.create(ExpressionConfig.builder()
			.modelPath(EXPRESSION_PATH.toString())
			.build())) {
			ExpressionResult result = recognizer.recognize(aligned);

			float[] reference = softmax(runExpressionOnnx(officialChw(
				aligned, new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f})));

			float[] actual = result.getProbabilities();
			assertThat(actual).hasSize(7);
			for (int i = 0; i < 7; i++) {
				assertThat(actual[i])
					.as("第 %d 类概率应与官方管线一致", i)
					.isCloseTo(reference[i], org.assertj.core.data.Offset.offset(1e-4f));
			}
			// 类别语义同样要对齐
			assertThat(result.getExpression().getIndex())
				.isEqualTo(indexOfMax(reference));
		} finally {
			aligned.release();
		}
	}

	/**
	 * 质量评估的生产预处理同样必须与官方管线逐位一致（与表情模型共用同一契约）。
	 */
	@Test
	void qualityPreprocessMatchesOfficialPipeline() throws OrtException {
		Mat aligned = buildColorfulAlignedFace();
		try (FaceQualityAssessor assessor = FaceQualityAssessor.create(FaceQualityConfig.builder()
			.modelPath(QUALITY_PATH.toString())
			.threshold(0.5f)
			.build())) {
			FaceQualityResult result = assessor.assess(aligned);

			float reference = runQualityOnnx(officialChw(
				aligned, new float[]{0.5f, 0.5f, 0.5f}, new float[]{0.5f, 0.5f, 0.5f}));

			assertThat(result.getScore()).isCloseTo(reference, org.assertj.core.data.Offset.offset(1e-4f));
		} finally {
			aligned.release();
		}
	}

	private static int indexOfMax(float[] values) {
		int idx = 0;
		for (int i = 1; i < values.length; i++) {
			if (values[i] > values[idx]) {
				idx = i;
			}
		}
		return idx;
	}
}
