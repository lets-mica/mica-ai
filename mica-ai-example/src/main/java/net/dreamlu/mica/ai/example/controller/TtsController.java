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
import net.dreamlu.mica.ai.tts.config.TtsResult;
import net.dreamlu.mica.ai.tts.engine.KokoroTts;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * 文本转语音（TTS）REST 端点。
 *
 * <p>仅在 {@code mica.ai.tts.enabled=true}（默认）时启用。
 */
@Tag(name = "TTS 语音合成", description = "Kokoro-82M · 中英 103 音色")
@RestController
@RequestMapping("/tts")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.tts", name = "enabled", havingValue = "true", matchIfMissing = true)
public class TtsController {

	private final KokoroTts tts;

	@Operation(summary = "列出可用音色", description = "返回 Kokoro-82M 支持的全部音色 ID（如 zf_001、zm_010 等）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "查询成功",
			content = @Content(mediaType = "application/json",
				schema = @Schema(implementation = List.class))),
		@ApiResponse(responseCode = "500", description = "模型未加载或异常", content = @Content)
	})
	@GetMapping("/voices")
	public List<String> listVoices() throws Exception {
		return tts.listVoices();
	}

	@Operation(summary = "文本合成语音", description = "传入文本，返回 WAV 字节流（Content-Type: audio/wav）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "合成成功，返回 audio/wav",
			content = @Content(mediaType = "audio/wav",
				schema = @Schema(type = "string", format = "binary"))),
		@ApiResponse(responseCode = "400", description = "参数非法（如 text 为空）", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/synthesize", produces = "audio/wav")
	public ResponseEntity<byte[]> synthesize(
		@Parameter(description = "待合成文本（中文或英文）", required = true, example = "你好，欢迎使用 mica-ai。")
		@RequestParam("text") String text,
		@Parameter(description = "音色名（可选，默认 zf_001）", example = "zf_001")
		@RequestParam(value = "voice", required = false) String voice,
		@Parameter(description = "语速 0.5 ~ 2.0（可选，默认 1.0）", example = "1.0")
		@RequestParam(value = "speed", required = false) Float speed) throws Exception {
		String v = voice == null || voice.isBlank() ? "zf_001" : voice;
		float s = speed == null ? 1.0f : speed;
		TtsResult result = tts.synthesize(text, v, s);
		byte[] wav = toWav(result);
		return ResponseEntity.ok()
			.header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"tts.wav\"")
			.contentType(MediaType.parseMediaType("audio/wav"))
			.body(wav);
	}

	@Operation(summary = "预生成音素合成语音", description = "用于绕过 G2P：直接使用预生成的音素序列合成 WAV")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "合成成功，返回 audio/wav",
			content = @Content(mediaType = "audio/wav",
				schema = @Schema(type = "string", format = "binary"))),
		@ApiResponse(responseCode = "400", description = "请求体缺少 phonemes 字段", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/synthesize-from-phonemes", produces = "audio/wav")
	public ResponseEntity<byte[]> synthesizeFromPhonemes(
		@Parameter(description = "请求体，字段: phonemes (音素串)、voice (音色，可选)、speed (语速，可选)", required = true)
		@RequestBody Map<String, Object> body) throws Exception {
		String phonemes = (String) body.get("phonemes");
		String voice = body.get("voice") == null ? "zf_001" : (String) body.get("voice");
		float speed = body.get("speed") == null ? 1.0f : ((Number) body.get("speed")).floatValue();
		TtsResult result = tts.synthesizeFromPhonemes(phonemes, voice, speed);
		byte[] wav = toWav(result);
		return ResponseEntity.ok()
			.contentType(MediaType.parseMediaType("audio/wav"))
			.body(wav);
	}

	private byte[] toWav(TtsResult result) throws Exception {
		java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
		try (java.io.DataOutputStream dos = new java.io.DataOutputStream(baos)) {
			int byteRate = result.sampleRate() * 2;
			int dataSize = result.audio().length * 2;
			dos.writeBytes("RIFF");
			dos.writeInt(Integer.reverseBytes(36 + dataSize));
			dos.writeBytes("WAVE");
			dos.writeBytes("fmt ");
			dos.writeInt(Integer.reverseBytes(16));
			dos.writeShort(Short.reverseBytes((short) 1));
			dos.writeShort(Short.reverseBytes((short) 1));
			dos.writeInt(Integer.reverseBytes(result.sampleRate()));
			dos.writeInt(Integer.reverseBytes(byteRate));
			dos.writeShort(Short.reverseBytes((short) 2));
			dos.writeShort(Short.reverseBytes((short) 16));
			dos.writeBytes("data");
			dos.writeInt(Integer.reverseBytes(dataSize));
			for (float sample : result.audio()) {
				float clamped = Math.max(-1f, Math.min(1f, sample));
				short pcm = (short) (clamped * 32767f);
				dos.writeShort(Short.reverseBytes(pcm));
			}
		}
		return baos.toByteArray();
	}
}
