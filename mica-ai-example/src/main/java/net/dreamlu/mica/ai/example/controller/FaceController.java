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
package net.dreamlu.mica.ai.example.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.face.alignment.FaceAligner;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.expression.ExpressionRecognizer;
import net.dreamlu.mica.ai.face.expression.ExpressionResult;
import net.dreamlu.mica.ai.face.liveness.LivenessDetector;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.LivenessResult;
import net.dreamlu.mica.ai.face.quality.FaceQualityAssessor;
import net.dreamlu.mica.ai.face.quality.FaceQualityResult;
import net.dreamlu.mica.ai.face.recognition.FeatureExtractor;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import net.dreamlu.mica.ai.face.verification.FaceVerifier;
import org.opencv.core.Mat;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 人脸检测 + 识别 REST 端点。
 */
@Tag(name = "Face 人脸识别", description = "OpenCV Zoo · YuNet 检测 + SFace 128d 向量")
@RestController
@RequestMapping("/face")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.face", name = "enabled", havingValue = "true", matchIfMissing = true)
public class FaceController {

	private final FaceDetector detector;
	private final FaceAligner aligner;
	private final FeatureExtractor extractor;
	private final FaceVerifier verifier;
	private final LivenessDetector livenessDetector;
	/**
	 * eDifFIQA 质量评估器：仅在 mica.ai.face.quality.enabled=true 时存在。
	 */
	private final ObjectProvider<FaceQualityAssessor> qualityAssessorProvider;
	/**
	 * 表情识别器：仅在 mica.ai.face.expression.enabled=true 时存在。
	 */
	private final ObjectProvider<ExpressionRecognizer> expressionRecognizerProvider;

	private static Path save(MultipartFile file) throws IOException {
		String original = file.getOriginalFilename() == null ? "face.png" : file.getOriginalFilename();
		Path tmp = Files.createTempFile("mica-face-", "-" + original);
		file.transferTo(tmp.toFile());
		tmp.toFile().deleteOnExit();
		return tmp;
	}

	private static Map<String, Object> boxView(FaceBox box) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("x1", box.getX1());
		m.put("y1", box.getY1());
		m.put("x2", box.getX2());
		m.put("y2", box.getY2());
		m.put("score", box.getScore());
		m.put("landmarks", box.getLandmarks());
		return m;
	}

	private static Map<String, Object> embeddingView(float[] vec) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("dim", vec.length);
		double norm = 0;
		for (float v : vec) {
			norm += (double) v * v;
		}
		m.put("l2Norm", Math.sqrt(norm));
		m.put("preview", preview(vec, 8));
		return m;
	}

	private static List<Float> preview(float[] vec, int n) {
		int len = Math.min(n, vec.length);
		List<Float> list = new java.util.ArrayList<>(len);
		for (int i = 0; i < len; i++) {
			list.add(vec[i]);
		}
		return list;
	}

	@Operation(summary = "人脸检测", description = "仅做检测：返回每张人脸的 bbox + 关键点 + score")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "检测成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/detect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public List<Map<String, Object>> detect(
		@Parameter(description = "待检测图片（支持 jpg/png/webp）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("file") MultipartFile file) throws IOException {
		Path tmp = save(file);
		Mat img = ImageUtils.byteArrayToMat(Files.readAllBytes(tmp));
		try {
			List<FaceBox> boxes = detector.detect(img);
			return boxes.stream().map(FaceController::boxView).collect(Collectors.toList());
		} finally {
			img.release();
			Files.deleteIfExists(tmp);
		}
	}

	@Operation(summary = "活体检测", description = "检测人脸后逐个做 MiniFASNetV2 活体判断，返回 liveScore / isLive / attackType")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "检测成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/liveness", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public List<Map<String, Object>> liveness(
		@Parameter(description = "待检测图片（支持 jpg/png/webp）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("file") MultipartFile file) throws IOException {
		Path tmp = save(file);
		Mat img = ImageUtils.byteArrayToMat(Files.readAllBytes(tmp));
		try {
			final Mat work = img;
			List<FaceBox> boxes = detector.detect(work);
			return boxes.stream().map(box -> livenessOne(work, box)).collect(Collectors.toList());
		} finally {
			img.release();
			Files.deleteIfExists(tmp);
		}
	}

	private Map<String, Object> livenessOne(Mat img, FaceBox box) {
		LivenessResult result = livenessDetector.check(img, box);
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("liveScore", result.getLiveScore());
		m.put("isLive", result.isLive());
		m.put("attackType", result.getAttackType());
		return m;
	}

	@Operation(summary = "人脸特征提取", description = "检测 + 对齐 + 提取 embedding，返回 128 维向量预览与 L2 范数")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "提取成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/extract", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public List<Map<String, Object>> extract(
		@Parameter(description = "待提取图片（支持 jpg/png/webp）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("file") MultipartFile file) throws IOException {
		Path tmp = save(file);
		Mat img = ImageUtils.byteArrayToMat(Files.readAllBytes(tmp));
		try {
			final Mat work = img;
			List<FaceBox> boxes = detector.detect(work);
			return boxes.stream().map(box -> extractOne(work, box)).collect(Collectors.toList());
		} finally {
			img.release();
			Files.deleteIfExists(tmp);
		}
	}

	private Map<String, Object> extractOne(Mat img, FaceBox box) {
		Mat aligned = null;
		try {
			aligned = aligner.align(img, box);
			float[] feature = extractor.extract(aligned);
			return embeddingView(feature);
		} finally {
			if (aligned != null) {
				aligned.release();
			}
		}
	}

	@Operation(summary = "人脸质量评估", description = "检测人脸后逐个做 eDifFIQA 质量评估，"
		+ "返回 score（越高越好）、threshold 与 passed；需要 mica.ai.face.quality.enabled=true")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "评估成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "503", description = "质量评估能力未启用", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/quality", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> quality(
		@Parameter(description = "待评估图片（支持 jpg/png/webp）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("file") MultipartFile file) throws IOException {
		FaceQualityAssessor assessor = qualityAssessorProvider.getIfAvailable();
		if (assessor == null) {
			return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
				.contentType(MediaType.APPLICATION_JSON)
				.body(Collections.singletonMap("error",
					"人脸质量评估未启用（mica.ai.face.quality.enabled=false）"));
		}
		Path tmp = save(file);
		Mat img = ImageUtils.byteArrayToMat(Files.readAllBytes(tmp));
		try {
			final Mat work = img;
			List<FaceBox> boxes = detector.detect(work);
			float threshold = assessor.getConfig().getThreshold();
			List<Map<String, Object>> results = new ArrayList<>(boxes.size());
			for (FaceBox box : boxes) {
				Mat aligned = null;
				try {
					aligned = aligner.align(work, box);
					FaceQualityResult result = assessor.assess(aligned);
					Map<String, Object> m = new LinkedHashMap<>();
					m.put("score", result.getScore());
					m.put("threshold", threshold);
					m.put("passed", result.passed(threshold));
					results.add(m);
				} finally {
					if (aligned != null) {
						aligned.release();
					}
				}
			}
			return ResponseEntity.ok(results);
		} finally {
			img.release();
			Files.deleteIfExists(tmp);
		}
	}

	@Operation(summary = "表情识别", description = "检测人脸后逐个识别 7 类表情"
		+ "（愤怒 / 厌恶 / 恐惧 / 开心 / 平静 / 悲伤 / 惊讶），返回各类概率；"
		+ "需要 mica.ai.face.expression.enabled=true")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "识别成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "503", description = "表情识别能力未启用", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/expression", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> expression(
		@Parameter(description = "待识别图片（支持 jpg/png/webp）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("file") MultipartFile file) throws IOException {
		ExpressionRecognizer recognizer = expressionRecognizerProvider.getIfAvailable();
		if (recognizer == null) {
			return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
				.contentType(MediaType.APPLICATION_JSON)
				.body(Collections.singletonMap("error",
					"表情识别未启用（mica.ai.face.expression.enabled=false）"));
		}
		Path tmp = save(file);
		Mat img = ImageUtils.byteArrayToMat(Files.readAllBytes(tmp));
		try {
			final Mat work = img;
			List<FaceBox> boxes = detector.detect(work);
			List<Map<String, Object>> results = new ArrayList<>(boxes.size());
			for (FaceBox box : boxes) {
				Mat aligned = null;
				try {
					aligned = aligner.align(work, box);
					ExpressionResult result = recognizer.recognize(aligned);
					Map<String, Object> m = new LinkedHashMap<>();
					m.put("expression", result.getExpression().name());
					m.put("desc", result.getExpression().getDesc());
					m.put("score", result.getScore());
					m.put("probabilities", result.getProbabilities());
					results.add(m);
				} finally {
					if (aligned != null) {
						aligned.release();
					}
				}
			}
			return ResponseEntity.ok(results);
		} finally {
			img.release();
			Files.deleteIfExists(tmp);
		}
	}
}
