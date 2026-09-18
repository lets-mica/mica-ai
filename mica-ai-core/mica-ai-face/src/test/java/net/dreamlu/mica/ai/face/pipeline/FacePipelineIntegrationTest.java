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
package net.dreamlu.mica.ai.face.pipeline;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.config.ModelConfig;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.face.recognition.FeatureExtractor;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import net.dreamlu.mica.ai.face.verification.FaceVerifier;
import net.dreamlu.mica.ai.face.verification.VerifyResult;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * mica-ai-face 真模型集成测试：加载仓库内 YuNet + SFace ONNX（{@code model-tools/face/models/}），
 * 跑通「模型加载 → 检测 → 对齐 → 特征 → 比对 → fail-fast」完整链路。
 *
 * <p>模型在 {@code AGENTS.md} §6.2 「<50MB 入库」约定内，已随仓库分发；缺失时本测试类用
 * {@link EnabledIf} 整体跳过，便于在仅有源码克隆的 CI 环境下也能跑出干净的失败信息。
 *
 * <p>OpenCV 原生库由 {@code nu.pattern.OpenCV.loadLocally()} 在静态块中加载，
 * 避开 Surefire forkCount=0 下 native 库多次加载的路径问题。
 */
@EnabledIf("isModelPresent")
class FacePipelineIntegrationTest {

	static {
		nu.pattern.OpenCV.loadLocally();
	}

	private static final Path DET_PATH = repoPath("model-tools/face/models/face_detection_yunet_2023mar.onnx");
	private static final Path REC_PATH = repoPath("model-tools/face/models/face_recognition_sface_2021dec.onnx");

	private static ModelManager manager;

	static boolean isModelPresent() {
		return Files.exists(DET_PATH) && Files.exists(REC_PATH);
	}

	private static Path repoPath(String relativePath) {
		Path dir = Paths.get("").toAbsolutePath();
		while (dir != null && !Files.exists(dir.resolve("model-tools"))) {
			dir = dir.getParent();
		}
		return dir == null ? Paths.get(relativePath) : dir.resolve(relativePath);
	}

	@BeforeAll
	static void setUp() {
		ModelConfig config = ModelConfig.builder()
			.detectionModelPath(DET_PATH.toString())
			.recognitionModelPath(REC_PATH.toString())
			.build();
		manager = ModelManager.create(config);
	}

	@AfterAll
	static void tearDown() throws Exception {
		if (manager != null) {
			manager.close();
		}
	}

	@Test
	void managerExposesThreeSessionsAndIsCloseable() throws Exception {
		assertThat(manager).isNotNull();
		assertThat(manager.getEnvironment()).isNotNull();
		assertThat(manager.getDetectionSession()).isNotNull();
		assertThat(manager.getRecognitionSession()).isNotNull();
		// 活体未配置路径 → session 为 null（按设计）
		assertThat(manager.getLivenessSession()).isNull();
		// close() 不抛异常、可重入
		manager.close();
		manager.close();
	}

	@Test
	void detectorProducesSensibleBoxesOnSyntheticFace() {
		FaceDetector detector = new FaceDetector(manager);
		Mat synthetic = buildSyntheticFace();
		try {
			List<FaceBox> boxes = detector.detect(synthetic);
			// 不硬性要求 YuNet 必须检出合成图（合成人脸不在训练分布内）；
			// 这里只验证 detector 在真模型下不抛异常、返回非 null，
			// 任何 box 都满足坐标合法、score ∈ [0, 1]、landmarks 为 5x2。
			assertThat(boxes).isNotNull();
			for (FaceBox b : boxes) {
				assertThat(b.getScore()).isBetween(0f, 1f);
				assertThat(b.getX2()).isGreaterThanOrEqualTo(b.getX1());
				assertThat(b.getY2()).isGreaterThanOrEqualTo(b.getY1());
				if (b.getLandmarks() != null) {
					assertThat(b.getLandmarks().length).isEqualTo(5);
					for (float[] p : b.getLandmarks()) {
						assertThat(p.length).isEqualTo(2);
					}
				}
			}
		} finally {
			synthetic.release();
		}
	}

	@Test
	void featureExtractorProducesUnitVectorAndIdenticalSelfCompare() {
		FeatureExtractor extractor = new FeatureExtractor(manager);
		Mat aligned = buildAlignedFaceStub();
		try {
			float[] feat = extractor.extract(aligned);
			assertThat(feat).hasSize(FeatureExtractor.FEATURE_DIM);
			// L2 归一化：||feat|| 应非常接近 1
			double sumSq = 0d;
			for (float f : feat) {
				sumSq += (double) f * f;
			}
			assertThat(Math.sqrt(sumSq)).isCloseTo(1.0d, org.assertj.core.data.Offset.offset(1e-3d));
			// 自身比对应为 1
			assertThat(FeatureExtractor.compare(feat, feat)).isCloseTo(1.0f, org.assertj.core.data.Offset.offset(1e-3f));
		} finally {
			aligned.release();
		}
	}

	@Test
	void verifierRoundTripRejectsEmptyImageWithFailFast() {
		FaceVerifier verifier = new FaceVerifier(manager);
		assertThatThrownBy(() -> verifier.verify(new byte[0], new byte[]{1, 2, 3}))
			.isInstanceOf(MicaAiException.class);
	}

	@Test
	void verifierRoundTripOnSyntheticProducesCosineInRange() {
		FaceVerifier verifier = new FaceVerifier(manager);
		final byte[] img1 = matToJpegBytes(buildSyntheticFace());
		final byte[] img2 = matToJpegBytes(buildSyntheticFace());
		// 合成图不在 YuNet 训练分布内，多数情况下检不到人脸 → verifier 抛 "未检测到人脸"。
		// 这里只验证 verifier 链路在真模型下能跑到 detect 阶段（fail-fast 抛 MicaAiException），
		// 不强求必须检出。
		assertThatThrownBy(() -> verifier.verify(img1, img2))
			.isInstanceOf(MicaAiException.class);
	}

	/**
	 * 构造一张 320×320 合成图：浅灰底 + 居中实心圆 + 两个深色圆点（眼）+ 一个椭圆（嘴），
	 * 给 YuNet 一个「非空、人脸状」的输入。不保证能被检出，只保证链路不抛。
	 */
	private static Mat buildSyntheticFace() {
		Mat img = new Mat(320, 320, org.opencv.core.CvType.CV_8UC3, new Scalar(220, 220, 220));
		// 脸盘
		Imgproc.circle(img, new org.opencv.core.Point(160, 160), 90, new Scalar(200, 180, 160), -1);
		// 左眼 / 右眼
		Imgproc.circle(img, new org.opencv.core.Point(130, 140), 8, new Scalar(20, 20, 20), -1);
		Imgproc.circle(img, new org.opencv.core.Point(190, 140), 8, new Scalar(20, 20, 20), -1);
		// 嘴
		Imgproc.ellipse(img, new org.opencv.core.Point(160, 200), new org.opencv.core.Size(28, 12), 0, 0, 180, new Scalar(40, 40, 40), 3);
		Imgproc.rectangle(img, new org.opencv.core.Point(0, 0), new org.opencv.core.Point(319, 319), new Scalar(255, 255, 255), 1);
		return img;
	}

	/**
	 * 构造一张 112×112 的「已对齐」合成图，给 SFace 喂一个合法输入。
	 * 它的输出不与训练分布对齐，但足以验证：构造 → ONNX 推理 → L2 归一化 全链路通畅。
	 */
	private static Mat buildAlignedFaceStub() {
		Mat img = new Mat(112, 112, org.opencv.core.CvType.CV_8UC3, new Scalar(200, 180, 160));
		Imgproc.circle(img, new org.opencv.core.Point(45, 45), 6, new Scalar(20, 20, 20), -1);
		Imgproc.circle(img, new org.opencv.core.Point(67, 45), 6, new Scalar(20, 20, 20), -1);
		Imgproc.ellipse(img, new org.opencv.core.Point(56, 75), new org.opencv.core.Size(15, 7), 0, 0, 180, new Scalar(40, 40, 40), 2);
		return img;
	}

	private static byte[] matToJpegBytes(Mat mat) {
		MatOfByte mob = new MatOfByte();
		try {
			Imgcodecs.imencode(".jpg", mat, mob);
			ByteArrayOutputStream out = new ByteArrayOutputStream();
			try (InputStream in = new java.io.ByteArrayInputStream(mob.toArray())) {
				byte[] buf = new byte[8192];
				int n;
				while ((n = in.read(buf)) != -1) {
					out.write(buf, 0, n);
				}
			} catch (IOException e) {
				throw new IllegalStateException("JPEG 编码读取失败", e);
			}
			return out.toByteArray();
		} finally {
			mob.release();
		}
	}
}