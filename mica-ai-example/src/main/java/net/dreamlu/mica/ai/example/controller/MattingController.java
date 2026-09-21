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
import net.dreamlu.mica.ai.matting.MattingEngine;
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
 * 通用抠图 REST 端点（U²-Net u2netp 显著性目标检测）。
 * <p>
 * 提供透明底 PNG（BGRA）、纯色底合成、二值掩码三组输出。所有可选参数未传时使用 starter 中
 * {@code mica.ai.matting.*} 配置的默认值。
 * </p>
 */
@Slf4j
@Tag(name = "Matting 通用抠图", description = "U²-Net u2netp · 透明底 PNG + 纯色底合成 + 二值掩码")
@RestController
@RequestMapping("/matting")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.matting", name = "enabled", havingValue = "true", matchIfMissing = true)
public class MattingController {

	private final MattingEngine mattingEngine;

	private static ResponseEntity<Object> image(byte[] bytes) {
		return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).body(bytes);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

	/**
	 * 抠图：输出带 alpha 通道的透明底 PNG。
	 */
	@Operation(summary = "抠图（透明底 PNG）",
		description = "输出 BGRA 四通道 PNG，背景透明；直接可用于前端叠加背景或合成")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "抠图成功（PNG 字节）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/cutout", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> cutout(
		@Parameter(description = "源图片（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			return image(mattingEngine.cutoutBytes(file.getBytes()));
		} catch (MicaAiException e) {
			log.warn("抠图失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	/**
	 * 抠图并合成纯色底：常用于证件照换底色（白 / 蓝 / 红）。
	 */
	@Operation(summary = "抠图并合成纯色底",
		description = "color 为 RGB 三元组（如 0,0,255 表示蓝底），未传时取 mica.ai.matting.background-color")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "合成成功（PNG 字节）"),
		@ApiResponse(responseCode = "400", description = "参数非法或图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/cutout-on-color", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> cutoutOnColor(
		@Parameter(description = "源图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "RGB 三元组，如 255,255,255（白底）/ 0,0,255（蓝底）/ 255,0,0（红底）")
		@RequestParam(value = "color", required = false) int[] color) throws IOException {
		try {
			return image(mattingEngine.cutoutOnColorBytes(file.getBytes(), color));
		} catch (MicaAiException e) {
			log.warn("纯色底合成失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	/**
	 * 二值掩码：输出单通道 0/255 PNG，适合需要硬边缘或作为后续处理的 mask 输入。
	 */
	@Operation(summary = "抠图（二值掩码）",
		description = "输出单通道掩码 PNG，前景 255 / 背景 0；threshold 未传时取 mica.ai.matting.binary-threshold")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "掩码生成成功（PNG 字节）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/mask", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> mask(
		@Parameter(description = "源图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			return image(mattingEngine.matteBinaryBytes(file.getBytes()));
		} catch (MicaAiException e) {
			log.warn("二值掩码生成失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}
}
