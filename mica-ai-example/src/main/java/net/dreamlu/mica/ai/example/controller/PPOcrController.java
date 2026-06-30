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
import net.dreamlu.mica.ai.ppocr.config.PPOcrV6Result;
import net.dreamlu.mica.ai.ppocr.engine.PPOcrV6Engine;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
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
 * OCR 文字识别 REST 端点。
 */
@Tag(name = "OCR 文字识别", description = "PP-OCRv6 · 检测 + 识别全链路")
@RestController
@RequestMapping("/ppocr")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.ppocr", name = "enabled", havingValue = "true", matchIfMissing = true)
public class PPOcrController {

	private final PPOcrV6Engine ocr;

	@Operation(summary = "图片文字识别", description = "上传图片，返回识别到的文本行（含位置和置信度）。底层走 PP-OCRv6 det+rec 全链路")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "识别成功",
			content = @Content(mediaType = "application/json",
				schema = @Schema(implementation = List.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public List<Map<String, Object>> recognize(
		@Parameter(description = "待识别图片（支持 jpg/png/webp/bmp）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("file") MultipartFile file) throws IOException {
		Path tmp = Files.createTempFile("mica-ppocr-", ".png");
		file.transferTo(tmp.toFile());
		tmp.toFile().deleteOnExit();
		try {
			Mat img = Imgcodecs.imread(tmp.toAbsolutePath().toString());
			if (img == null || img.empty()) {
				throw new IllegalArgumentException("无法读取图片: " + file.getOriginalFilename());
			}
			List<PPOcrV6Result> results = ocr.run(img);
			img.release();
			return results.stream().map(PPOcrController::toView).toList();
		} finally {
			Files.deleteIfExists(tmp);
		}
	}

	private static Map<String, Object> toView(PPOcrV6Result r) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("text", r.text());
		m.put("score", r.score());
		m.put("box", r.boxAsNestedList());
		return m;
	}
}
