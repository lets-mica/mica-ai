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
import net.dreamlu.mica.ai.speaker.engine.SpeakerVerifier;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
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
@Tag(name = "Speaker 声纹识别", description = "ERes2NetV2 · 192d Embedding · 验证 / 识别")
@RestController
@RequestMapping("/speaker")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.speaker", name = "enabled", havingValue = "true", matchIfMissing = true)
public class SpeakerController {

	private final SpeakerVerifier verifier;

	@Operation(summary = "声纹注册", description = "上传多段同一人的音频，平均成一个 192 维声纹 embedding（仅返回预览）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "注册成功",
			content = @Content(mediaType = "application/json",
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "音频文件为空或读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/enroll", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> enroll(
		@Parameter(description = "注册音频列表（wav 格式，可多段同一人）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("files") List<MultipartFile> files) throws IOException {
		List<Path> wavs = saveAll(files);
		try {
			float[] emb = verifier.enrollSpeaker(wavs);
			return embeddingView("enrolled", emb);
		} finally {
			wavs.forEach(p -> p.toFile().delete());
		}
	}

	@Operation(summary = "1:1 声纹验证", description = "对比 uploaded 声纹 embedding（base64 或 hex）与 test wav。本 Demo 仅演示已注册声纹 + 验证音频模式（不做 embedding 持久化）。")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "验证成功",
			content = @Content(mediaType = "application/json",
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "音频文件为空或读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/verify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public Map<String, Object> verify(
		@Parameter(description = "注册音频（wav 格式）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("enroll") MultipartFile enroll,
		@Parameter(description = "待验证音频（wav 格式）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
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