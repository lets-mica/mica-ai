/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example.controller;

import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.face.config.FaceBox;
import net.dreamlu.mica.ai.face.config.FaceEmbedding;
import net.dreamlu.mica.ai.face.engine.FaceEngine;
import net.dreamlu.mica.ai.face.utils.ImageUtils;
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
 * 人脸检测 + 识别 REST 端点。
 */
@RestController
@RequestMapping("/face")
@RequiredArgsConstructor
@ConditionalOnBean(FaceEngine.class)
public class FaceController {

	private final FaceEngine face;

	/**
	 * 仅做检测：返回每张人脸的 bbox + 关键点 + score。
	 */
	@PostMapping("/detect")
	public List<Map<String, Object>> detect(@RequestParam("file") MultipartFile file) throws IOException {
		Path tmp = save(file);
		try {
			var img = ImageUtils.read(tmp);
			List<FaceBox> boxes = face.detect(img);
			return boxes.stream().map(FaceController::boxView).toList();
		} finally {
			Files.deleteIfExists(tmp);
		}
	}

	/**
	 * 检测 + 提取 embedding。
	 */
	@PostMapping("/extract")
	public List<Map<String, Object>> extract(@RequestParam("file") MultipartFile file) throws IOException {
		Path tmp = save(file);
		try {
			var img = ImageUtils.read(tmp);
			List<FaceEmbedding> embeddings = face.extract(img);
			return embeddings.stream().map(FaceController::embeddingView).toList();
		} finally {
			Files.deleteIfExists(tmp);
		}
	}

	private static Path save(MultipartFile file) throws IOException {
		String original = file.getOriginalFilename() == null ? "face.png" : file.getOriginalFilename();
		Path tmp = Files.createTempFile("mica-face-", "-" + original);
		file.transferTo(tmp.toFile());
		tmp.toFile().deleteOnExit();
		return tmp;
	}

	private static Map<String, Object> boxView(FaceBox box) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("x1", box.getX1());
		m.put("y1", box.getY1());
		m.put("x2", box.getX2());
		m.put("y2", box.getY2());
		m.put("score", box.getScore());
		m.put("landmarks", box.getLandmarks());
		return m;
	}

	private static Map<String, Object> embeddingView(FaceEmbedding emb) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("dim", emb.dimension());
		double norm = 0;
		for (float v : emb.getVector()) {
			norm += (double) v * v;
		}
		m.put("l2Norm", Math.sqrt(norm));
		m.put("preview", preview(emb.getVector(), 8));
		return m;
	}

	private static List<Float> preview(float[] vec, int n) {
		int len = Math.min(n, vec.length);
		List<Float> list = new java.util.ArrayList<>(len);
		for (int i = 0; i < len; i++) {
			list.add(vec[i]);
		}
		return list;
	}
}