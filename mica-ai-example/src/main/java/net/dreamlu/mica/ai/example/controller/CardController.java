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
import net.dreamlu.mica.ai.face.card.CardExtractor;
import net.dreamlu.mica.ai.face.card.CardOptions;
import net.dreamlu.mica.ai.face.card.CardResult;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.*;

/**
 * 证件卡片提取端点。
 * <p>
 * 提供单图最高分卡、单图全部候选两个端点；所有可选参数未传时使用 starter 中
 * {@code mica.ai.face.card.*} 配置的默认值。
 * </p>
 */
@Slf4j
@Tag(name = "Face 证件卡片提取", description = "证件照片定位 + 透视矫正 + 锐化")
@RestController
@RequestMapping("/face/card")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.face", name = "enabled", havingValue = "true", matchIfMissing = true)
public class CardController {

	private final CardExtractor cardExtractor;

	private static int quality(Integer quality) {
		return quality == null ? 95 : quality;
	}

	private static MediaType contentType(String format) {
		return "jpg".equalsIgnoreCase(format) || "jpeg".equalsIgnoreCase(format)
			? MediaType.IMAGE_JPEG : MediaType.IMAGE_PNG;
	}

	private static ResponseEntity<Object> image(byte[] bytes, MediaType type,
												Map<String, String> headers) {
		ResponseEntity.BodyBuilder builder = ResponseEntity.ok().contentType(type);
		if (headers != null) {
			headers.forEach(builder::header);
		}
		return builder.body(bytes);
	}

	private static ResponseEntity<Object> json(Object body, HttpStatus status) {
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> failure(MicaAiException e) {
		Map<String, Object> body = error(e.getMessage());
		return json(body, HttpStatus.BAD_REQUEST);
	}

	private static void releaseAll(List<CardResult> results) {
		if (results != null) {
			results.forEach(CardResult::release);
		}
	}

	private static Map<String, Object> error(String msg) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return m;
	}

	/**
	 * 提取卡片：返回画面中得分最高的卡片，直接输出图片字节。
	 */
	@Operation(summary = "提取卡片（最高分）",
		description = "返回画面中得分最高的卡片图，Content-Type 由 format 决定；若未检测到卡片返回 400")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "提取成功（图片字节）"),
		@ApiResponse(responseCode = "400", description = "未检测到卡片", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/extract", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> extract(
		@Parameter(description = "源图片（身份证 / 护照 / 银行卡等证件照片）", required = true,
			schema = @Schema(type = "string", format = "binary"))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "输出格式 png/jpg，默认 png") @RequestParam(value = "format", required = false) String format,
		@Parameter(description = "JPEG 质量 [1,100]，默认 95") @RequestParam(value = "quality", required = false) Integer quality)
		throws IOException {

		String fmt = format == null ? "png" : format;
		Mat image = null;
		List<CardResult> results = null;
		try {
			image = ImageUtils.byteArrayToMat(file.getBytes());
			results = cardExtractor.extractAll(image, CardOptions.defaults());
			if (results.isEmpty()) {
				return json(error("未检测到卡片"), HttpStatus.BAD_REQUEST);
			}
			CardResult first = results.get(0);
			Map<String, String> headers = new LinkedHashMap<>();
			headers.put("X-Card-Score", String.format("%.3f", first.getScore()));
			headers.put("X-Card-Aspect", String.format("%.3f", first.getAspectRatio()));
			headers.put("X-Card-Size", String.format("%.1f", first.getCardSize()));
			headers.put("X-Card-Sharpness", String.format("%.1f", first.getSharpness()));
			headers.put("X-Card-Usable", String.valueOf(first.isUsable()));
			headers.put("X-Card-Orientation", String.valueOf(first.getRotationDegrees()));
			headers.put("X-Card-Auto-Oriented", String.valueOf(first.isAutoOriented()));
			return image(ImageUtils.matToBytes(first.getImage(), fmt, quality(quality)),
				contentType(fmt), headers);
		} catch (MicaAiException e) {
			log.warn("卡片提取失败: {}", e.getMessage());
			return failure(e);
		} finally {
			releaseAll(results);
			ImageUtils.releaseAll(image);
		}
	}

	/**
	 * 提取全部候选卡片：返回每张卡片的元数据与 base64 图片，按得分降序。
	 */
	@Operation(summary = "提取卡片（全部候选）",
		description = "返回检测到的每张卡片（最多 maxCards 张）的元数据（四角、长宽比、得分、清晰度、朝向）与 base64 图片，按得分降序")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "提取成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/cards", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Map<String, Object>> cards(
		@Parameter(description = "源图片", required = true,
			schema = @Schema(type = "string", format = "binary"))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "输出格式 png/jpg") @RequestParam(value = "format", required = false) String format,
		@Parameter(description = "JPEG 质量 [1,100]") @RequestParam(value = "quality", required = false) Integer quality,
		@Parameter(description = "最多返回卡片数，0 为不限") @RequestParam(value = "maxCards", required = false) Integer maxCards)
		throws IOException {

		String fmt = format == null ? "png" : format;
		Mat image = null;
		List<CardResult> results = null;
		try {
			image = ImageUtils.byteArrayToMat(file.getBytes());
			results = cardExtractor.extractAll(image, CardOptions.defaults());
			int limit = maxCards == null ? 0 : maxCards;
			if (limit > 0 && results.size() > limit) {
				// 释放被截断的尾部，避免 Mat 泄漏
				for (int i = limit; i < results.size(); i++) {
					results.get(i).release();
				}
				results = new ArrayList<>(results.subList(0, limit));
			}

			List<Map<String, Object>> cardList = new ArrayList<>(results.size());
			for (CardResult r : results) {
				Map<String, Object> cm = new LinkedHashMap<>();
				cm.put("index", r.getIndex());
				cm.put("score", r.getScore());
				cm.put("aspectRatio", r.getAspectRatio());
				cm.put("quad", r.getQuad());
				cm.put("cardSize", r.getCardSize());
				cm.put("usable", r.isUsable());
				cm.put("unusableReason", r.getUnusableReason());
				cm.put("sharpness", r.getSharpness());
				cm.put("rotationDegrees", r.getRotationDegrees());
				cm.put("autoOriented", r.isAutoOriented());
				cm.put("format", fmt);
				cm.put("imageBase64", Base64.getEncoder()
					.encodeToString(ImageUtils.matToBytes(r.getImage(), fmt, quality(quality))));
				cardList.add(cm);
			}
			Map<String, Object> ok = new LinkedHashMap<>();
			ok.put("count", cardList.size());
			ok.put("cards", cardList);
			return ResponseEntity.ok(ok);
		} catch (MicaAiException e) {
			log.warn("卡片提取失败: {}", e.getMessage());
			Map<String, Object> body = error(e.getMessage());
			return ResponseEntity.badRequest().body(body);
		} finally {
			releaseAll(results);
			ImageUtils.releaseAll(image);
		}
	}
}
