/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example.controller;

import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.tts.config.TtsResult;
import net.dreamlu.mica.ai.tts.engine.KokoroTts;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/**
 * 文本转语音（TTS）REST 端点。
 *
 * <p>仅在 {@link KokoroTts} Bean 存在时启用。
 */
@RestController
@RequestMapping("/tts")
@RequiredArgsConstructor
@ConditionalOnBean(KokoroTts.class)
public class TtsController {

	private final KokoroTts tts;

	/**
	 * 列出可用音色。
	 */
	@GetMapping("/voices")
	public List<String> listVoices() throws Exception {
		return tts.listVoices();
	}

	/**
	 * 文本合成语音并直接返回 WAV 字节流。
	 *
	 * @param text  待合成文本
	 * @param voice 音色名（可选）
	 * @param speed 语速 0.5 ~ 2.0（可选）
	 */
	@PostMapping(value = "/synthesize", produces = "audio/wav")
	public ResponseEntity<byte[]> synthesize(@RequestParam("text") String text,
											 @RequestParam(value = "voice", required = false) String voice,
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

	/**
	 * 用预生成音素合成语音（用于绕过 G2P）。
	 */
	@PostMapping(value = "/synthesize-from-phonemes", produces = "audio/wav")
	public ResponseEntity<byte[]> synthesizeFromPhonemes(@RequestBody Map<String, Object> body) throws Exception {
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