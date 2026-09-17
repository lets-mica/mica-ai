/*
 * Copyright (c) 2024-2026 mica-ai
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
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.example.pipeline.FacePipeline;
import net.dreamlu.mica.ai.example.pipeline.FaceRecognitionResult;
import net.dreamlu.mica.ai.example.repository.VectorRepository;
import net.dreamlu.mica.ai.face.autoconfigure.FaceProperties;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.LivenessResult;
import net.dreamlu.mica.ai.face.model.MatchResult;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.face.recognition.FeatureExtractor;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import net.dreamlu.mica.ai.face.verification.FaceVerifier;
import net.dreamlu.mica.ai.face.verification.VerifyResult;
import org.opencv.core.Mat;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.util.*;

/**
 * 人脸注册 / 1:N 识别 / 1:1 比对（人证核验）/ 健康检查端点。
 * <p>
 * 依赖 {@link FacePipeline} 与 {@link VectorRepository}（见 {@code FaceRecognitionConfig}）。
 * </p>
 */
@Slf4j
@Tag(name = "Face 人脸服务", description = "人脸注册、1:N 识别、1:1 比对（人证核验）与健康检查")
@RestController
@RequestMapping("/face")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.face", name = "enabled", havingValue = "true", matchIfMissing = true)
public class FaceRecognitionController {

	private final FacePipeline facePipeline;
	private final VectorRepository vectorRepository;
	private final ModelManager modelManager;
	private final FaceProperties properties;
	private final FaceVerifier faceVerifier;
	private final FaceDetector detector;

	private static Map<String, Object> error(String msg) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return m;
	}

	/**
	 * 注册人脸：上传图片 + personId，提取第一张人脸特征后存入向量仓库。
	 */
	@Operation(summary = "注册人脸", description = "上传图片 + personId，提取第一张人脸特征后存入向量仓库")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "注册成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "未检测到人脸 / 活体未通过 / personId 为空", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/register", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> register(
		@Parameter(description = "人脸图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "人员唯一标识", required = true, example = "user-001")
		@RequestParam("personId") String personId) throws IOException {
		if (personId == null || personId.isEmpty()) {
			throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "personId 不能为空");
		}
		Mat mat = null;
		try {
			mat = ImageUtils.byteArrayToMat(file.getBytes());
			FaceRecognitionResult result = facePipeline.recognize(mat);
			if (result.getFaces().isEmpty()) {
				throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "未检测到人脸");
			}
			float[] feature = result.getFeatures().get(0);
			if (feature == null) {
				throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "活体检测未通过，无法注册");
			}
			vectorRepository.save(personId, feature);
			Map<String, Object> ok = new LinkedHashMap<>();
			ok.put("personId", personId);
			ok.put("status", "registered");
			ok.put("score", result.getFaces().get(0).getScore());
			return ok;
		} catch (MicaAiException e) {
			log.warn("注册失败: {}", e.getMessage());
			throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, e.getMessage());
		} finally {
			ImageUtils.releaseAll(mat);
		}
	}

	/**
	 * 1:N 人脸识别。
	 */
	@Operation(summary = "1:N 人脸识别", description = "上传图片，返回检测到的人脸框、活体结果与匹配人员")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "识别成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> recognize(
		@Parameter(description = "待识别人脸图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		Mat mat = null;
		try {
			mat = ImageUtils.byteArrayToMat(file.getBytes());
			FaceRecognitionResult result = facePipeline.recognize(mat);
			List<Map<String, Object>> faceList = new ArrayList<>();
			for (int i = 0; i < result.getFaces().size(); i++) {
				FaceBox box = result.getFaces().get(i);
				Map<String, Object> fm = new LinkedHashMap<>();
				fm.put("index", i);
				fm.put("score", box.getScore());
				fm.put("bbox", new float[]{box.getX1(), box.getY1(), box.getX2(), box.getY2()});
				LivenessResult lr = result.getLivenessMap().get(i);
				fm.put("liveness", lr);
				List<MatchResult> matches = result.getMatches().get(i);
				fm.put("matches", matches != null ? matches : Collections.emptyList());
				fm.put("matched", matches != null && !matches.isEmpty());
				faceList.add(fm);
			}
			Map<String, Object> ok = new LinkedHashMap<>();
			ok.put("faces", faceList);
			ok.put("count", faceList.size());
			return ok;
		} catch (MicaAiException e) {
			log.warn("识别失败: {}", e.getMessage());
			throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, e.getMessage());
		} finally {
			ImageUtils.releaseAll(mat);
		}
	}

	/**
	 * 1:1 人脸比对（人证核验）。
	 */
	@Operation(summary = "1:1 人脸比对（人证核验）", description = "上传 probe（现场）+ reference（证件照），返回相似度与是否通过")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "比对完成",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/verify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> verify(
		@Parameter(description = "现场采集人脸", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("probe") MultipartFile probe,
		@Parameter(description = "参考人脸（如证件照）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("reference") MultipartFile reference,
		@Parameter(description = "相似度阈值（可选，默认取 mica.ai.face.verify.threshold）")
		@RequestParam(value = "threshold", required = false) Float threshold) throws IOException {
		try {
			VerifyResult vr = faceVerifier.verify(probe.getBytes(), reference.getBytes(), threshold);
			Map<String, Object> ok = new LinkedHashMap<>();
			ok.put("similarity", vr.getSimilarity());
			ok.put("threshold", vr.getThreshold());
			ok.put("passed", vr.isPassed());
			return ok;
		} catch (MicaAiException e) {
			log.warn("1:1 比对失败: {}", e.getMessage());
			throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, e.getMessage());
		}
	}

	/**
	 * 人证核验快捷方式：先取底库特征，再与现场图 1:1 比对。
	 */
	@Operation(summary = "人证核验（底库特征 + 现场图 1:1）",
		description = "上传 probe + personId：用 personId 从底库取特征，再与现场人脸比对")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "核验完成",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "personId 未注册 / 未检测到人脸 / 活体未通过", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/identity-verify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> identityVerify(
		@Parameter(description = "现场采集人脸", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("probe") MultipartFile probe,
		@Parameter(description = "证件 / 底库中的 personId", required = true)
		@RequestParam("personId") String personId,
		@Parameter(description = "相似度阈值（可选）")
		@RequestParam(value = "threshold", required = false) Float threshold) throws IOException {
		float[] ref = vectorRepository.getFeature(personId);
		if (ref == null) {
			throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "personId 未注册: " + personId);
		}
		Mat mat = null;
		try {
			mat = ImageUtils.byteArrayToMat(probe.getBytes());
			float[] probeFeat = extractLargest(mat);
			float sim = FeatureExtractor.compare(probeFeat, ref);
			float th = threshold != null ? threshold : properties.getVerify().getThreshold();
			Map<String, Object> ok = new LinkedHashMap<>();
			ok.put("personId", personId);
			ok.put("similarity", sim);
			ok.put("threshold", th);
			ok.put("passed", sim >= th);
			return ok;
		} catch (MicaAiException e) {
			log.warn("人证核验失败: {}", e.getMessage());
			throw new ResponseStatusException(HttpStatus.INTERNAL_SERVER_ERROR, e.getMessage());
		} finally {
			ImageUtils.releaseAll(mat);
		}
	}

	/**
	 * 健康检查：返回活体开关与模型加载状态。
	 */
	@Operation(summary = "健康检查", description = "返回活体开关与模型加载状态")
	@GetMapping("/health")
	public Map<String, Object> health() {
		Map<String, Object> r = new LinkedHashMap<>();
		r.put("livenessEnabled", properties.getLiveness().isEnabled());
		r.put("verifyThreshold", properties.getVerify().getThreshold());
		r.put("detectionLoaded", modelManager.getDetectionSession() != null);
		r.put("recognitionLoaded", modelManager.getRecognitionSession() != null);
		r.put("livenessLoaded", modelManager.getLivenessSession() != null);
		r.put("onnxOptions", properties.getOnnx());
		return r;
	}

	// 复用 pipeline 的对齐与提取，从最大脸取特征
	private float[] extractLargest(Mat image) {
		FaceRecognitionResult result = facePipeline.recognize(image);
		if (result.getFaces().isEmpty()) {
			throw new MicaAiException(ErrorCode.VERIFICATION_FAILED, "未检测到人脸");
		}
		int idx = 0;
		float bestArea = area(result.getFaces().get(0));
		for (int i = 1; i < result.getFaces().size(); i++) {
			float a = area(result.getFaces().get(i));
			if (a > bestArea) {
				bestArea = a;
				idx = i;
			}
		}
		float[] feat = result.getFeatures().get(idx);
		if (feat == null) {
			throw new MicaAiException(ErrorCode.VERIFICATION_FAILED, "活体检测未通过");
		}
		return feat;
	}

	private static float area(FaceBox box) {
		return (box.getX2() - box.getX1()) * (box.getY2() - box.getY1());
	}
}
