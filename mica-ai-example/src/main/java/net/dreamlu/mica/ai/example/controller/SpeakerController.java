/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example.controller;

import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.speaker.engine.SpeakerVerifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 声纹识别 REST 端点。
 */
@RestController
@RequestMapping("/speaker")
@RequiredArgsConstructor
@ConditionalOnBean(SpeakerVerifier.class)
public class SpeakerController {

	private final SpeakerVerifier verifier;

	/**
	 * 多段注册音频平均成一个声纹（192 维）。
	 */
	@PostMapping("/enroll")
	public Map<String, Object> enroll(@RequestParam("files") List<MultipartFile> files) throws IOException {
		List<Path> wavs = saveAll(files);
		try {
			float[] emb = verifier.enrollSpeaker(wavs);
			return embeddingView("enrolled", emb);
		} finally {
			wavs.forEach(p -> p.toFile().delete());
		}
	}

	/**
	 * 1:1 验证：对比 uploaded 声纹 embedding（base64 或 hex）与 test wav。
	 *
	 * <p>本 Demo 仅演示"已注册声纹 + 验证音频"模式（不做 embedding 持久化）。
	 */
	@PostMapping("/verify")
	public Map<String, Object> verify(@RequestParam("enroll") MultipartFile enroll,
									   @RequestParam("test") MultipartFile test) throws IOException {
		Path enrollWav = save(enroll);
		Path testWav = save(test);
		try {
			float[] enrolled = verifier.extractEmbedding(enrollWav);
			float score = verifier.verify(enrolled, testWav)
				? Float.POSITIVE_INFINITY
				: -Float.POSITIVE_INFINITY;
			float rawScore = SpeakerVerifier.cosineSimilarity(
				enrolled, verifier.extractEmbedding(testWav));
			Map<String, Object> view = new LinkedHashMap<>();
			view.put("matched", rawScore >= SpeakerVerifier.DEFAULT_THRESHOLD);
			view.put("score", rawScore);
			view.put("threshold", SpeakerVerifier.DEFAULT_THRESHOLD);
			return view;
		} finally {
			enrollWav.toFile().delete();
			testWav.toFile().delete();
		}
	}

	private static Path save(MultipartFile file) throws IOException {
		Path tmp = Files.createTempFile("mica-speaker-", ".wav");
		file.transferTo(tmp.toFile());
		tmp.toFile().deleteOnExit();
		return tmp;
	}

	private static List<Path> saveAll(List<MultipartFile> files) throws IOException {
		return files.stream().map(f -> {
			try {
				return save(f);
			} catch (IOException e) {
				throw new RuntimeException(e);
			}
		}).toList();
	}

	private static Map<String, Object> embeddingView(String label, float[] emb) {
		Map<String, Object> view = new LinkedHashMap<>();
		view.put("label", label);
		view.put("dim", emb.length);
		double norm = 0;
		for (float v : emb) {
			norm += (double) v * v;
		}
		view.put("l2Norm", Math.sqrt(norm));
		view.put("preview", preview(emb, 8));
		return view;
	}

	private static List<Float> preview(float[] emb, int n) {
		int len = Math.min(n, emb.length);
		List<Float> list = new java.util.ArrayList<>(len);
		for (int i = 0; i < len; i++) {
			list.add(emb[i]);
		}
		return list;
	}
}