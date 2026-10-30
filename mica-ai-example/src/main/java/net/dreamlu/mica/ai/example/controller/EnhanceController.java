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
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.example.pipeline.enhance.EnhanceUpscaleDetail;
import net.dreamlu.mica.ai.example.pipeline.enhance.EnhanceUpscaleOptions;
import net.dreamlu.mica.ai.example.pipeline.enhance.EnhanceUpscalePipeline;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 图像超分 REST 端点（Real-ESRGAN realesr-general-x4v3）。
 * <p>
 * 4 倍放大，适合抓拍修复、缩略图放大、证件照清晰化。
 * 大图会自动切 tile 推理（{@code mica.ai.enhance.tile-size} / {@code tile-overlap}），
 * 避免一次性把超大图塞进显存。
 * </p>
 */
@Slf4j
@Tag(name = "Enhance 图像超分", description = "Real-ESRGAN · 4 倍放大（tile 推理，无接缝）")
@RestController
@RequestMapping("/enhance")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.enhance", name = "enabled", havingValue = "true", matchIfMissing = true)
public class EnhanceController {

	private final EnhanceEngine enhanceEngine;

	private final EnhanceUpscalePipeline upscalePipeline;

	private static ResponseEntity<Object> image(byte[] bytes) {
		return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).body(bytes);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

	/**
	 * 图片放大：输出放大后的 PNG。
	 */
	@Operation(summary = "图像超分放大",
		description = "按 mica.ai.enhance.scale（默认 4 倍）放大图片并输出 PNG；"
			+ "大图自动切 tile 推理，重叠上下文用于消除拼接接缝")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "放大成功（PNG 字节）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/upscale", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> upscale(
		@Parameter(description = "待放大的图片（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			return image(enhanceEngine.upscaleBytes(file.getBytes()));
		} catch (MicaAiException e) {
			log.warn("图像超分失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	/**
	 * 图像超分（带对比信息）。
	 *
	 * <p>返回 JSON：同时内联「送入模型的原图」与「超分输出」两张 PNG（base64），
	 * 供前端做同源对比滑块；再附尺寸、倍率、耗时等元信息。
	 *
	 * <p>输出倍率小于模型原生倍率时，先按原生倍率推理再重采样回目标倍率；
	 * 输入长边超过 {@code maxInputSide} 时先等比缩小，避免超大图打爆内存与响应体。
	 */
	@Operation(summary = "图像超分（带原图对比）",
		description = "输出 JSON：originalBase64 为送入模型的原图、outputBase64 为超分结果，"
			+ "两者同源可直接做对比滑块；outScale 可选 1~4（默认取模型原生倍率），"
			+ "tileSize 控制分块大小（越小越省内存越慢），maxInputSide 为输入长边上限")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "超分成功（含对比图与元信息）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败或参数非法", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/upscale-detail", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> upscaleDetail(
		@Parameter(description = "待放大的图片（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "输出倍率，1 ~ 模型原生倍率（默认取模型原生倍率）")
		@RequestParam(value = "outScale", required = false) Integer outScale,
		@Parameter(description = "分块边长，32 ~ 2048（默认 256）")
		@RequestParam(value = "tileSize", required = false) Integer tileSize,
		@Parameter(description = "输入长边上限，超过先等比缩小（默认 1200）")
		@RequestParam(value = "maxInputSide", required = false) Integer maxInputSide) {
		try {
			EnhanceUpscaleOptions defaults = EnhanceUpscaleOptions.builder().build();
			EnhanceUpscaleOptions options = EnhanceUpscaleOptions.builder()
				.outScale(outScale == null ? upscalePipeline.modelScale() : outScale)
				.tileSize(tileSize == null ? defaults.getTileSize() : tileSize)
				.maxInputSide(maxInputSide == null ? defaults.getMaxInputSide() : maxInputSide)
				.build();
			EnhanceUpscaleDetail detail = new EnhanceUpscaleDetail(
				upscalePipeline.upscale(file.getBytes(), options));
			return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(detail);
		} catch (MicaAiException e) {
			log.warn("图像超分失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		} catch (IOException e) {
			log.warn("图像超分失败: 图片读取异常", e);
			return error("图片读取失败: " + e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}
}
