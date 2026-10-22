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
import net.dreamlu.mica.ai.captcha.CaptchaEngine;
import net.dreamlu.mica.ai.captcha.model.CaptchaResult;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
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
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 通用验证码识别 REST 端点（ddddocr CRNN + CTC）。
 * <p>
 * 输入是<b>单行文字为主体的验证码图片</b>，输出识别文本与置信度。
 * 模型与字符集为外置配置，由 {@code mica.ai.captcha.model-path} /
 * {@code mica.ai.captcha.charset-path} 指定。
 * </p>
 */
@Slf4j
@Tag(name = "Captcha 通用验证码识别", description = "ddddocr CRNN + CTC · 单行验证码文本与置信度")
@RestController
@RequestMapping("/captcha")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.captcha", name = "enabled", havingValue = "true", matchIfMissing = true)
public class CaptchaController {

	private final CaptchaEngine captchaEngine;

	private static ResponseEntity<Object> json(Map<String, Object> body) {
		return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

	/**
	 * 识别验证码：返回文本与置信度。
	 */
	@Operation(summary = "识别验证码",
		description = "上传验证码图片，返回 text（识别文本）、confidence（平均置信度）、"
			+ "charConfidences（逐字符置信度）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "识别成功（JSON）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> recognize(
		@Parameter(description = "验证码图片（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			CaptchaResult r = captchaEngine.recognizeDetailed(file.getBytes());
			Map<String, Object> body = new LinkedHashMap<>();
			body.put("text", r.getText());
			body.put("confidence", r.getConfidence());
			body.put("charConfidences", r.getCharConfidences());
			return json(body);
		} catch (MicaAiException e) {
			log.warn("验证码识别失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}
}
