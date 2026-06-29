/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example.controller;

import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.intent.config.IntentResult;
import net.dreamlu.mica.ai.intent.engine.BertIntent;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 中文意图分类 REST 端点。
 */
@RestController
@RequestMapping("/intent")
@RequiredArgsConstructor
@ConditionalOnBean(BertIntent.class)
public class IntentController {

	private final BertIntent intent;

	/**
	 * 单条预测。
	 */
	@PostMapping("/predict")
	public Map<String, Object> predict(@RequestBody Map<String, String> body) {
		IntentResult result = intent.predict(body.get("text"));
		return toView(result);
	}

	/**
	 * 批量预测。
	 */
	@PostMapping("/predict-batch")
	public List<Map<String, Object>> predictBatch(@RequestBody Map<String, List<String>> body) {
		List<IntentResult> results = intent.predictBatch(body.get("texts"));
		return results.stream().map(IntentController::toView).toList();
	}

	private static Map<String, Object> toView(IntentResult result) {
		Map<String, Object> view = new LinkedHashMap<>();
		view.put("intent", result.intent());
		view.put("confidence", result.confidence());
		view.put("allScores", result.allScores());
		return view;
	}
}