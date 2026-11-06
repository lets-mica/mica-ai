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
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.barcode.core.BarcodeEngine;
import net.dreamlu.mica.ai.barcode.model.BarcodeResult;
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
import java.util.Optional;

/**
 * 二维码 / 一维条码解码 REST 端点（ZXing core，零模型零 ONNX）。
 *
 * <p>{@code /decode} 走全量扫描找齐一图多码；{@code /decode-first} 命中即短路，单码场景更快。
 */
@Slf4j
@Tag(name = "Barcode 二维码 / 一维条码", description = "ZXing core · 多尺度 + 分块重试 · 零模型")
@RestController
@RequestMapping("/barcode")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.barcode", name = "enabled", havingValue = "true", matchIfMissing = true)
public class BarcodeController {

	private final BarcodeEngine barcodeEngine;

	@Operation(summary = "解码图中的所有条码",
		description = "上传票据 / 面单 / 资产标签图片，返回去重后的全部码（文本、制式、近似包围盒、命中策略）")
	@PostMapping(value = "/decode", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> decode(
		@Parameter(description = "包含二维码 / 一维条码的图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			List<BarcodeResult> results = barcodeEngine.decodeBytes(file.getBytes());
			List<Map<String, Object>> codes = new ArrayList<>();
			for (BarcodeResult result : results) {
				codes.add(toMap(result));
			}
			Map<String, Object> body = new LinkedHashMap<>();
			body.put("count", codes.size());
			body.put("codes", codes);
			return json(body);
		} catch (IllegalArgumentException e) {
			log.warn("条码解码失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	@Operation(summary = "只解第一个条码",
		description = "命中即短路，不做全量分块扫描；未命中时 found=false")
	@PostMapping(value = "/decode-first", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> decodeFirst(
		@Parameter(description = "包含二维码 / 一维条码的图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			Optional<BarcodeResult> result = barcodeEngine.decodeFirstBytes(file.getBytes());
			Map<String, Object> body = new LinkedHashMap<>();
			body.put("found", result.isPresent());
			body.put("code", result.map(BarcodeController::toMap).orElse(null));
			return json(body);
		} catch (IllegalArgumentException e) {
			log.warn("条码解码失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	private static Map<String, Object> toMap(BarcodeResult result) {
		Map<String, Object> item = new LinkedHashMap<>();
		item.put("text", result.getText());
		item.put("symbology", result.getSymbology().name());
		item.put("charset", result.getCharset());
		item.put("boundingBox", result.getBoundingBox());
		item.put("strategy", result.getStrategy());
		return item;
	}

	private static ResponseEntity<Object> json(Map<String, Object> body) {
		return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> body = new LinkedHashMap<>();
		body.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(body);
	}
}
