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
import net.dreamlu.mica.ai.voice.config.RecognitionResult;
import net.dreamlu.mica.ai.voice.config.TranscriptionResult;
import net.dreamlu.mica.ai.voice.engine.SenseVoice;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 语音识别（ASR）REST 端点。
 */
@Tag(name = "ASR 语音识别", description = "SenseVoiceSmall · 多语种 + Trie 树热词雷达")
@RestController
@RequestMapping("/voice")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.voice", name = "enabled", havingValue = "true", matchIfMissing = true)
public class VoiceController {

	private final SenseVoice voice;

	@Operation(summary = "语音识别", description = "上传 WAV 文件，返回识别文本、热词命中、分段时间（frontendMs/encoderMs/decoderMs/radarMs/integrateMs/totalMs）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "识别成功",
			content = @Content(mediaType = "application/json",
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "音频文件为空或读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> recognize(
		@Parameter(description = "待识别音频（wav 格式）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("file") MultipartFile file) throws Exception {
		TranscriptionResult result = voice.recognizeFile(toPath(file).toString());
		return toView(result);
	}

	@Operation(summary = "动态更新热词", description = "替换 SenseVoice 内置的 Trie 树热词列表（立即生效）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "更新成功",
			content = @Content(mediaType = "application/json",
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "请求体为空", content = @Content),
		@ApiResponse(responseCode = "500", description = "更新异常", content = @Content)
	})
	@PutMapping("/hotwords")
	public Map<String, Object> updateHotwords(
		@Parameter(description = "热词字符串数组（用于提升识别命中率）", required = true)
		@RequestBody List<String> hotwords) {
		voice.updateHotwords(hotwords);
		return Map.of("updated", true, "size", hotwords.size());
	}

	private static java.nio.file.Path toPath(MultipartFile file) throws java.io.IOException {
		java.nio.file.Path tmp = java.nio.file.Files.createTempFile("mica-voice-", ".wav");
		file.transferTo(tmp.toFile());
		tmp.toFile().deleteOnExit();
		return tmp;
	}

	private static Map<String, Object> toView(TranscriptionResult result) {
		Map<String, Object> view = new LinkedHashMap<>();
		view.put("text", result.text());
		view.put("hotWords", result.hotWords());
		view.put("timings", Map.of(
			"frontendMs", result.timings().frontend() * 1000,
			"encoderMs", result.timings().encoder() * 1000,
			"decoderMs", result.timings().decoder() * 1000,
			"radarMs", result.timings().radar() * 1000,
			"integrateMs", result.timings().integrate() * 1000,
			"totalMs", result.timings().total() * 1000
		));
		view.put("results", result.results().stream().map(VoiceController::toView).toList());
		return view;
	}

	private static Map<String, Object> toView(RecognitionResult r) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("text", r.text());
		m.put("startSec", r.start());
		m.put("hotWord", r.hotWord());
		return m;
	}
}