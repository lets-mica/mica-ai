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
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.textline.TextLineEngine;
import net.dreamlu.mica.ai.textline.model.TextLineOrientationResult;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 文本行方向分类 REST 端点（PP-LCNet_x1_0_textline_ori）。
 * <p>
 * 输入是<b>已裁剪好的单行文本图</b>（不是整页文档），输出该行是 0 度还是 180 度，
 * 以及把该行转正所需的旋转角度。典型用法是 OCR 流水线的前置转正。
 * </p>
 * <p>
 * 判定阈值取 starter 中 {@code mica.ai.textline.upside-down-threshold} 配置。
 * </p>
 */
@Slf4j
@Tag(name = "TextLine 文本行方向分类", description = "PP-LCNet_x1_0_textline_ori · 0 度 / 180 度判定与转正")
@RestController
@RequestMapping("/textline")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.textline", name = "enabled", havingValue = "true", matchIfMissing = true)
public class TextLineController {

	private final TextLineEngine textlineEngine;

	private static ResponseEntity<Object> image(byte[] bytes) {
		return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).body(bytes);
	}

	private static ResponseEntity<Object> json(Map<String, Object> body) {
		return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

	/**
	 * 判定文本行方向：返回方向枚举、置信度与转正角度。
	 */
	@Operation(summary = "判定文本行方向",
		description = "输入单行文本图，返回 orientation（0_degree / 180_degree）、score（置信度）、"
			+ "upsideDown（是否倒置）、angle（转正所需角度 0 / 180）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "判定成功（JSON）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/classify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> classify(
		@Parameter(description = "单行文本图（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			TextLineOrientationResult r = textlineEngine.classifyBytes(file.getBytes());
			if (r == null) {
				return error("图像为空或无法解码", HttpStatus.BAD_REQUEST);
			}
			Map<String, Object> body = new LinkedHashMap<>();
			body.put("orientation", r.getOrientation().name());
			body.put("label", r.getOrientation().getLabel());
			body.put("score", r.getScore());
			body.put("upsideDown", r.isUpsideDown());
			body.put("angle", r.angle());
			return json(body);
		} catch (MicaAiException e) {
			log.warn("文本行方向判定失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	/**
	 * 转正文本行：倒置则旋转 180 度输出 PNG，方向正常则原样返回输入图片。
	 */
	@Operation(summary = "转正文本行（倒置才旋转）",
		description = "判定为倒置时旋转 180 度后输出 PNG；方向正常时原样回传输入图片字节")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "转正成功（PNG 或原始图片字节）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/upright", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> upright(
		@Parameter(description = "单行文本图", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			return image(textlineEngine.uprightBytes(file.getBytes()));
		} catch (MicaAiException e) {
			log.warn("文本行转正失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}
}
