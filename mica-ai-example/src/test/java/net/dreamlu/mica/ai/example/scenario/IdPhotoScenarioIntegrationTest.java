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

import net.dreamlu.mica.ai.enhance.config.EnhanceConfig;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoBackground;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoMaskSource;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoOptions;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoPipeline;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoResult;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoSpec;
import net.dreamlu.mica.ai.face.alignment.FaceAligner;
import net.dreamlu.mica.ai.face.config.FaceQualityConfig;
import net.dreamlu.mica.ai.face.config.ModelConfig;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.face.quality.FaceQualityAssessor;
import net.dreamlu.mica.ai.matting.MattingEngine;
import net.dreamlu.mica.ai.matting.config.MattingConfig;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * 证件照一键生成集成测试。
 *
 * <p>模型缺失时整类跳过，不阻塞新克隆仓库的构建。
 */
class IdPhotoScenarioIntegrationTest {

	private static IdPhotoPipeline pipeline;

	@BeforeAll
	static void setUp() {
		nu.pattern.OpenCV.loadLocally();

		Path detectionModel = repoFile(
			"model-tools/face/models/face_detection_yunet_2023mar.onnx");
		Path recognitionModel = repoFile(
			"model-tools/face/models/face_recognition_sface_2021dec.onnx");
		Path qualityModel = repoFile("model-tools/face/models/ediffiqa_tiny_jun2024.onnx");
		Path mattingModel = repoFile("model-tools/matting/models/u2netp.onnx");
		Path enhanceModel = repoFile("model-tools/enhance/models/realesr-general-x4v3.onnx");
		if (detectionModel == null || recognitionModel == null || qualityModel == null
			|| mattingModel == null || enhanceModel == null) {
			return;
		}

		ModelManager modelManager = new ModelManager(ModelConfig.builder()
			.detectionModelPath(detectionModel.toAbsolutePath().toString())
			.recognitionModelPath(recognitionModel.toAbsolutePath().toString())
			// 测试素材里的人脸较小（最高分约 0.88），默认阈值 0.9 会漏检，这里放宽到 0.8
			.detectionThreshold(0.8f)
			.nmsThreshold(0.3f)
			.build());
		FaceDetector detector = new FaceDetector(modelManager);
		FaceAligner aligner = new FaceAligner();
		FaceQualityAssessor quality = FaceQualityAssessor.create(FaceQualityConfig.builder()
			.modelPath(qualityModel.toAbsolutePath().toString()).build());
		MattingEngine matting = MattingEngine.create(MattingConfig.builder()
			.modelPath(mattingModel.toAbsolutePath().toString()).build());
		EnhanceEngine enhance = EnhanceEngine.create(EnhanceConfig.builder()
			.modelPath(enhanceModel.toAbsolutePath().toString()).build());

		pipeline = new IdPhotoPipeline(detector, aligner, quality, matting, null, enhance);
	}

	@Test
	@DisplayName("尺寸规格：毫米按 300 DPI 换算成像素")
	void specShouldConvertMillimetersToPixels() {
		assertThat(IdPhotoSpec.ONE.widthPx(300)).isEqualTo(295);
		assertThat(IdPhotoSpec.ONE.heightPx(300)).isEqualTo(413);
		assertThat(IdPhotoSpec.SMALL_ONE.widthPx(300)).isEqualTo(260);
		assertThat(IdPhotoSpec.SMALL_ONE.heightPx(300)).isEqualTo(378);
		assertThat(IdPhotoSpec.LARGE_ONE.widthPx(300)).isEqualTo(390);
		assertThat(IdPhotoSpec.LARGE_ONE.heightPx(300)).isEqualTo(567);
		assertThat(IdPhotoSpec.TWO.widthPx(300)).isEqualTo(413);
		assertThat(IdPhotoSpec.TWO.heightPx(300)).isEqualTo(579);
		assertThat(IdPhotoSpec.of("ONE")).isEqualTo(IdPhotoSpec.ONE);
		assertThat(IdPhotoSpec.of("一寸")).isEqualTo(IdPhotoSpec.ONE);
		assertThat(IdPhotoSpec.of("不存在")).isNull();
	}

	@Test
	@DisplayName("底色与掩码来源：按名称解析，RGB 入参")
	void backgroundShouldExposeRgb() {
		assertThat(IdPhotoBackground.of("BLUE").rgb()).containsExactly(67, 142, 219);
		assertThat(IdPhotoBackground.of("白底")).isEqualTo(IdPhotoBackground.WHITE);
		assertThat(IdPhotoBackground.of("不存在")).isNull();
		assertThat(IdPhotoMaskSource.of("MATTING")).isEqualTo(IdPhotoMaskSource.MATTING);
		assertThat(IdPhotoMaskSource.of("不存在")).isNull();
	}

	@Test
	@DisplayName("无人脸图片应判定为 NO_FACE 且不产出成片")
	void shouldReportNoFaceForBlankImage() {
		assumeReady();
		IdPhotoResult result = pipeline.process(blankImage(),
			IdPhotoOptions.builder().qualityGate(false).upscale(false).build());
		assertThat(result.isPassed()).isFalse();
		assertThat(result.getReason()).isEqualTo("NO_FACE");
		assertThat(result.getOutputPng()).isNull();
		assertThat(result.getHint()).contains("人脸");
	}

	@Test
	@DisplayName("成片几何：头顶留白 7%、头高占比 75%、越界用底色补齐")
	void composeShouldPlaceHeadAccordingToSpec() {
		// 造一张 400x600 的底图，中间放一个矩形当「头」。
		// 几何按流水线公式反推，保证与实现自洽：
		//   脸 = [160,200]-[240,260]（脸高 60）
		//   头高 = 脸高 × 2 = 120 => 头 = [160,170]-[240,290]（头顶比脸顶高 30 = 脸高/2）
		int imgWidth = 400;
		int imgHeight = 600;
		int headX = 160;
		int headY = 170;
		int headW = 80;
		int headH = 120;
		org.opencv.core.Mat bgr = new org.opencv.core.Mat(imgHeight, imgWidth,
			org.opencv.core.CvType.CV_8UC3, new org.opencv.core.Scalar(30, 30, 30));
		org.opencv.core.Mat alpha = new org.opencv.core.Mat(imgHeight, imgWidth,
			org.opencv.core.CvType.CV_32FC1, new org.opencv.core.Scalar(0, 0, 0));
		try {
			org.opencv.core.Mat mask = new org.opencv.core.Mat(imgHeight, imgWidth,
				org.opencv.core.CvType.CV_8UC1, new org.opencv.core.Scalar(0, 0, 0));
			mask.submat(headY, headY + headH, headX, headX + headW)
				.setTo(new org.opencv.core.Scalar(255, 255, 255));
			bgr.setTo(new org.opencv.core.Scalar(10, 10, 200), mask);
			alpha.setTo(new org.opencv.core.Scalar(1, 1, 1), mask);
			mask.release();

			IdPhotoOptions options = IdPhotoOptions.builder()
				.spec(IdPhotoSpec.ONE)
				.background(IdPhotoBackground.WHITE)
				.dpi(300)
				.build();
			float[] faceBox = new float[]{160, 200, 240, 260};

			int[] crop = IdPhotoPipeline.computeCropRect(faceBox, options);
			org.opencv.core.Mat out = IdPhotoPipeline.composeIdPhoto(bgr, alpha, faceBox, options);
			try {
				// 一寸 @300dpi
				assertThat(out.cols()).isEqualTo(295);
				assertThat(out.rows()).isEqualTo(413);

				// 头顶留白 7% => 头顶在第 round(413 * 0.07) = 29 行附近
				// 头高占比 75% => 头占 round(413 * 0.75) = 310 行
				int topMargin = Math.round(413 * options.getTopMarginRatio());
				int headHeight = Math.round(413 * options.getHeadHeightRatio());
				assertThat(topMargin).isEqualTo(29);
				assertThat(headHeight).isEqualTo(310);

				// 头顶上方是背景（alpha=0）=> 必须是纯白底色
				assertThat(pixel(out, topMargin - 5, out.cols() / 2))
					.containsExactly(255, 255, 255);
				// 头部中央（alpha=1）=> 必须保留原图像素（BGR 10,10,200）
				assertThat(pixel(out, topMargin + headHeight / 2, out.cols() / 2))
					.containsExactly(10, 10, 200);
			} finally {
				out.release();
			}

			// 裁剪框宽高比应与规格一致（25:35）
			int cropWidth = crop[2] - crop[0];
			int cropHeight = crop[3] - crop[1];
			assertThat((float) cropWidth / cropHeight).isCloseTo(25f / 35f, org.assertj.core.data.Offset.offset(0.02f));
		} finally {
			bgr.release();
			alpha.release();
		}
	}

	@Test
	@DisplayName("成片几何：越界部分用底色补齐而不是拉伸")
	void composeShouldPadOutOfBoundsWithBackground() {
		int imgWidth = 120;
		int imgHeight = 120;
		org.opencv.core.Mat bgr = new org.opencv.core.Mat(imgHeight, imgWidth,
			org.opencv.core.CvType.CV_8UC3, new org.opencv.core.Scalar(30, 30, 30));
		org.opencv.core.Mat alpha = new org.opencv.core.Mat(imgHeight, imgWidth,
			org.opencv.core.CvType.CV_32FC1, new org.opencv.core.Scalar(0, 0, 0));
		try {
			IdPhotoOptions options = IdPhotoOptions.builder()
				.spec(IdPhotoSpec.ONE)
				.background(IdPhotoBackground.BLUE)
				.dpi(300)
				.build();
			// 人脸框贴着左上角 => 裁剪框会大幅超出原图
			float[] faceBox = new float[]{10, 10, 50, 50};
			org.opencv.core.Mat out = IdPhotoPipeline.composeIdPhoto(bgr, alpha, faceBox, options);
			try {
				assertThat(out.cols()).isEqualTo(295);
				assertThat(out.rows()).isEqualTo(413);
				// 原图只有 120x120，裁剪框必然越界 => 底色应铺满大部分画面
				assertThat(pixel(out, 5, 5)).containsExactly(219, 142, 67);
				assertThat(pixel(out, 408, 290)).containsExactly(219, 142, 67);
			} finally {
				out.release();
			}
		} finally {
			bgr.release();
			alpha.release();
		}
	}

	/**
	 * 读取某个像素的 BGR 三通道值。
	 */
	private static int[] pixel(org.opencv.core.Mat mat, int row, int col) {
		double[] p = mat.get(row, col);
		return new int[]{(int) Math.round(p[0]), (int) Math.round(p[1]), (int) Math.round(p[2])};
	}

	@Test
	@DisplayName("人像照片应产出符合规格的证件照")
	void shouldProduceIdPhotoForFaceImage() throws IOException {
		assumeReady();
		Path photo = repoFile(
			"mica-ai-core/mica-ai-classify/src/test/resources/person/someone.jpg");
		assumeTrue(photo != null, "人像测试素材缺失，跳过");

		IdPhotoOptions options = IdPhotoOptions.builder()
			.spec(IdPhotoSpec.ONE)
			.background(IdPhotoBackground.WHITE)
			.dpi(300)
			.maskSource(IdPhotoMaskSource.MATTING)
			.qualityGate(false)
			.upscale(false)
			.build();
		IdPhotoResult result = pipeline.process(Files.readAllBytes(photo), options);

		assertThat(result.isPassed()).isTrue();
		assertThat(result.getReason()).isEqualTo("OK");
		assertThat(result.getOutputPng()).isNotEmpty();
		// 一寸 @300dpi = 295x413
		assertThat(result.getWidth()).isEqualTo(295);
		assertThat(result.getHeight()).isEqualTo(413);
		assertThat(result.getFaceBox()).isNotNull();
		assertThat(result.getCropBox()).isNotNull();
		assertThat(result.getSpec()).isEqualTo("一寸");
		assertThat(result.getBackground()).isEqualTo("白底");
		// 裁剪框必须落在原图范围内（该素材人脸较小，裁剪框仍在图内）
		assertThat(result.getCropBox()[0]).isGreaterThanOrEqualTo(0);
		assertThat(result.getCropBox()[3]).isLessThanOrEqualTo(1024);
	}

	@Test
	@DisplayName("超分开关：开启后成片放大 4 倍供印刷")
	void shouldUpscaleIdPhotoForPrint() throws IOException {
		assumeReady();
		Path photo = repoFile(
			"mica-ai-core/mica-ai-classify/src/test/resources/person/someone.jpg");
		assumeTrue(photo != null, "人像测试素材缺失，跳过");

		IdPhotoOptions options = IdPhotoOptions.builder()
			.spec(IdPhotoSpec.ONE)
			.qualityGate(false)
			.upscale(true)
			.build();
		IdPhotoResult result = pipeline.process(Files.readAllBytes(photo), options);

		assertThat(result.isPassed()).isTrue();
		assertThat(result.isUpscaled()).isTrue();
		assertThat(result.getUpscaleScale()).isEqualTo(4);
		assertThat(result.getWidth()).isEqualTo(295 * 4);
		assertThat(result.getHeight()).isEqualTo(413 * 4);
	}

	@Test
	@DisplayName("质量门控默认值应为 false：评估照做但不拦截出片")
	void qualityGateShouldDefaultToOff() {
		assertThat(IdPhotoOptions.builder().build().isQualityGate()).isFalse();
		assertThat(IdPhotoOptions.builder().build().getMinQuality()).isEqualTo(0.5f);
	}

	@Test
	@DisplayName("质量门控开启且不达标：判 LOW_QUALITY 且不产出成片")
	void shouldRejectLowQualityWhenGateEnabled() throws IOException {
		assumeReady();
		Path photo = repoFile(
			"mica-ai-core/mica-ai-classify/src/test/resources/person/someone.jpg");
		assumeTrue(photo != null, "人像测试素材缺失，跳过");

		// 门限取 2.0，超出 eDifFIQA 取值范围 [0,1]，必然判不达标
		IdPhotoResult result = pipeline.process(Files.readAllBytes(photo),
			IdPhotoOptions.builder()
				.qualityGate(true)
				.minQuality(2.0f)
				.upscale(false)
				.build());

		assertThat(result.isPassed()).isFalse();
		assertThat(result.getReason()).isEqualTo("LOW_QUALITY");
		assertThat(result.getOutputPng()).isNull();
		assertThat(result.getQualityPassed()).isFalse();
		assertThat(result.getHint()).contains("质量分");
	}

	@Test
	@DisplayName("同样的低质量，门控关闭：仍然出片且回填质量分与不达标结论")
	void shouldStillProduceWhenGateDisabled() throws IOException {
		assumeReady();
		Path photo = repoFile(
			"mica-ai-core/mica-ai-classify/src/test/resources/person/someone.jpg");
		assumeTrue(photo != null, "人像测试素材缺失，跳过");

		IdPhotoResult result = pipeline.process(Files.readAllBytes(photo),
			IdPhotoOptions.builder()
				.qualityGate(false)
				.minQuality(2.0f)
				.upscale(false)
				.build());

		assertThat(result.isPassed()).isTrue();
		assertThat(result.getReason()).isEqualTo("OK");
		assertThat(result.getOutputPng()).isNotEmpty();
		// 评估照做：不达标结论回填，但只是提示，不拦片
		assertThat(result.getQualityPassed()).isFalse();
		assertThat(result.getQualityScore()).isGreaterThanOrEqualTo(0f);
		assertThat(result.getHint()).contains("未开启质量门控");
	}

	private static void assumeReady() {
		assumeTrue(pipeline != null, "相关模型缺失，跳过证件照集成测试");
	}

	/**
	 * 生成一张纯白图（无人物）。
	 */
	private static byte[] blankImage() {
		org.opencv.core.Mat mat = new org.opencv.core.Mat(360, 480,
			org.opencv.core.CvType.CV_8UC3,
			new org.opencv.core.Scalar(255, 255, 255));
		try {
			return net.dreamlu.mica.ai.matting.util.MattingImageUtils.imencodePng(mat);
		} finally {
			mat.release();
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
