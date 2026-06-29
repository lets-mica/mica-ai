/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example.controller;

import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.ppocr.config.PPOcrV6Result;
import net.dreamlu.mica.ai.ppocr.engine.PPOcrV6Engine;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
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
 * OCR 文字识别 REST 端点。
 */
@RestController
@RequestMapping("/ppocr")
@RequiredArgsConstructor
@ConditionalOnBean(PPOcrV6Engine.class)
public class PPOcrController {

	private final PPOcrV6Engine ocr;

	/**
	 * 上传图片，返回识别到的文本行（含位置和置信度）。
	 */
	@PostMapping("/recognize")
	public List<Map<String, Object>> recognize(@RequestParam("file") MultipartFile file) throws IOException {
		Path tmp = Files.createTempFile("mica-ppocr-", ".png");
		file.transferTo(tmp.toFile());
		tmp.toFile().deleteOnExit();
		try {
			Mat img = Imgcodecs.imread(tmp.toAbsolutePath().toString());
			if (img == null || img.empty()) {
				throw new IllegalArgumentException("无法读取图片: " + file.getOriginalFilename());
			}
			List<PPOcrV6Result> results = ocr.run(img);
			img.release();
			return results.stream().map(PPOcrController::toView).toList();
		} finally {
			Files.deleteIfExists(tmp);
		}
	}

	private static Map<String, Object> toView(PPOcrV6Result r) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("text", r.text());
		m.put("score", r.score());
		m.put("box", r.boxAsNestedList());
		return m;
	}
}
