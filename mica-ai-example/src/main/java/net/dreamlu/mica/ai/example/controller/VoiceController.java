/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example.controller;

import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.voice.config.RecognitionResult;
import net.dreamlu.mica.ai.voice.config.TranscriptionResult;
import net.dreamlu.mica.ai.voice.engine.SenseVoice;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
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
@RestController
@RequestMapping("/voice")
@RequiredArgsConstructor
@ConditionalOnBean(SenseVoice.class)
public class VoiceController {

	private final SenseVoice voice;

	/**
	 * 识别上传的 WAV 文件（multipart/form-data）。
	 */
	@PostMapping("/recognize")
	public Map<String, Object> recognize(@RequestParam("file") MultipartFile file) throws Exception {
		TranscriptionResult result = voice.recognizeFile(toPath(file).toString());
		return toView(result);
	}

	/**
	 * 动态更新热词列表。
	 */
	@PutMapping("/hotwords")
	public Map<String, Object> updateHotwords(@RequestBody List<String> hotwords) {
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