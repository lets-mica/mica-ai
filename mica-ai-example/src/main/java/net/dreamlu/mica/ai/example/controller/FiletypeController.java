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
import net.dreamlu.mica.ai.filetype.detection.FiletypeDetector;
import net.dreamlu.mica.ai.filetype.model.ContentTypeInfo;
import net.dreamlu.mica.ai.filetype.model.FiletypeResult;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 文件类型识别 REST 端点（Google Magika standard_v3_3 Java 复刻）。
 */
@Tag(name = "Filetype 文件类型识别", description = "Google Magika · 214 类 + content_types_kb 知识库")
@RestController
@RequestMapping("/filetype")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.filetype", name = "enabled", havingValue = "true", matchIfMissing = true)
public class FiletypeController {

	private final FiletypeDetector detector;

	@Operation(summary = "上传文件识别", description = "识别上传文件的类型，返回 model label / output label / score / mime / group / description")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "识别成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/detect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> detectUpload(
		@Parameter(description = "待识别任意文件", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try (InputStream in = file.getInputStream()) {
			return view(detector.detectStream(in));
		}
	}

	@Operation(summary = "原始字节识别", description = "上传任意二进制字节数组进行识别（适合集成方直接传 Buffer / byte[]）")
	@PostMapping(value = "/detect-bytes", consumes = MediaType.APPLICATION_OCTET_STREAM_VALUE)
	public Map<String, Object> detectBytes(
		@Parameter(description = "任意二进制内容", required = true,
			content = @Content(mediaType = MediaType.APPLICATION_OCTET_STREAM_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("file") MultipartFile file) throws IOException {
		return view(detector.detectBytes(file.getBytes()));
	}

	private static Map<String, Object> view(FiletypeResult result) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("outputLabel", result.getOutputLabel());
		m.put("modelLabel", result.getModelLabel());
		m.put("score", result.getScore());
		m.put("text", result.isText());
		m.put("mode", result.getMode().name());
		ContentTypeInfo ct = result.getContentType();
		if (ct != null) {
			m.put("mimeType", ct.getMimeType());
			m.put("group", ct.getGroup());
			m.put("description", ct.getDescription());
			m.put("extensions", ct.getExtensions());
		}
		return m;
	}
}
