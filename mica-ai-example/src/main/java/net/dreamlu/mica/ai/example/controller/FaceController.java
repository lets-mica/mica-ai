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
import net.dreamlu.mica.ai.face.config.FaceBox;
import net.dreamlu.mica.ai.face.config.FaceEmbedding;
import net.dreamlu.mica.ai.face.engine.FaceEngine;
import net.dreamlu.mica.ai.face.utils.ImageUtils;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 人脸检测 + 识别 REST 端点。
 */
@Tag(name = "Face 人脸识别", description = "OpenCV Zoo · YuNet 检测 + SFace 512d 向量")
@RestController
@RequestMapping("/face")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.face", name = "enabled", havingValue = "true", matchIfMissing = true)
public class FaceController {

	private final FaceEngine face;

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
		try {
			var img = ImageUtils.read(tmp);
			List<FaceBox> boxes = face.detect(img);
			return boxes.stream().map(FaceController::boxView).toList();
		} finally {
			Files.deleteIfExists(tmp);
		}
	}

	@Operation(summary = "人脸特征提取", description = "检测 + 提取 embedding，返回 512 维向量预览与 L2 范数")
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
		try {
			var img = ImageUtils.read(tmp);
			List<FaceEmbedding> embeddings = face.extract(img);
			return embeddings.stream().map(FaceController::embeddingView).toList();
		} finally {
			Files.deleteIfExists(tmp);
		}
	}

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

	private static Map<String, Object> embeddingView(FaceEmbedding emb) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("dim", emb.dimension());
		double norm = 0;
		for (float v : emb.getVector()) {
			norm += (double) v * v;
		}
		m.put("l2Norm", Math.sqrt(norm));
		m.put("preview", preview(emb.getVector(), 8));
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
}