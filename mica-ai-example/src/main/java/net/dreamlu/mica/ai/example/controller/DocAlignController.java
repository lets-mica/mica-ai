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
import net.dreamlu.mica.ai.docalign.core.DocAlignEngine;
import net.dreamlu.mica.ai.docalign.model.DocAlignResult;
import net.dreamlu.mica.ai.docalign.model.DocCorner;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 文档角点检测 / 透视矫正 REST 端点（DocAligner）。
 * <p>
 * 典型流水线：手机拍文档 → 角点检测 → 透视拉平 → 交给版面分析 / OCR。
 * 拉平结果再喂给 {@code /face/card/extract} 效果更好（先摆正整图，再裁卡片）。
 * </p>
 */
@Slf4j
@Tag(name = "DocAlign 文档角点矫正", description = "DocAligner · 4 角点检测 + 透视拉平")
@RestController
@RequestMapping("/docalign")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.docalign", name = "enabled", havingValue = "true", matchIfMissing = true)
public class DocAlignController {

	private final DocAlignEngine docAlignEngine;

	private static ResponseEntity<Object> json(Map<String, Object> body) {
		return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> image(byte[] bytes) {
		return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).body(bytes);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

	private static Map<String, Object> cornerView(String name, DocCorner corner) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("name", name);
		m.put("x", corner.getX());
		m.put("y", corner.getY());
		m.put("score", corner.getScore());
		return m;
	}

	/**
	 * 检测文档四角。
	 */
	@Operation(summary = "检测文档四角",
		description = "返回左上 / 右上 / 右下 / 左下四个角点的原图坐标与置信度（score 偏低说明该角点未可靠命中）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "检测成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/corners", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> corners(
		@Parameter(description = "拍摄或扫描的文档图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			DocAlignResult result = docAlignEngine.cornersBytes(file.getBytes());
			List<Map<String, Object>> corners = new ArrayList<>(DocAlignResult.CORNER_COUNT);
			corners.add(cornerView("topLeft", result.getTopLeft()));
			corners.add(cornerView("topRight", result.getTopRight()));
			corners.add(cornerView("bottomRight", result.getBottomRight()));
			corners.add(cornerView("bottomLeft", result.getBottomLeft()));

			Map<String, Object> body = new LinkedHashMap<>();
			body.put("width", result.getWidth());
			body.put("height", result.getHeight());
			body.put("corners", corners);
			return json(body);
		} catch (MicaAiException e) {
			log.warn("文档角点检测失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	/**
	 * 检测角点并透视拉平，输出 PNG。
	 */
	@Operation(summary = "文档透视矫正（拉平）",
		description = "一步完成「检测四角 + 透视矫正」，输出拉平后的 PNG，输出尺寸按角点边长自动推算")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "矫正成功（PNG 字节）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/align", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> align(
		@Parameter(description = "拍摄或扫描的文档图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			return image(docAlignEngine.alignBytes(file.getBytes()));
		} catch (MicaAiException e) {
			log.warn("文档透视矫正失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}
}
