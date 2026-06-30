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
import net.dreamlu.mica.ai.intent.config.IntentResult;
import net.dreamlu.mica.ai.intent.engine.BertIntent;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
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
@Tag(name = "Intent 意图识别", description = "BERT 中文意图分类 · Softmax 分类头")
@RestController
@RequestMapping("/intent")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.intent", name = "enabled", havingValue = "true", matchIfMissing = true)
public class IntentController {

	private final BertIntent intent;

	@Operation(summary = "单条意图预测", description = "传入单条中文文本，返回 Top-1 意图标签与置信度，及全部意图的得分")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "预测成功",
			content = @Content(mediaType = "application/json",
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "请求体缺少 text 字段", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping("/predict")
	public Map<String, Object> predict(
		@Parameter(description = "请求体，字段: text (中文文本)", required = true)
		@RequestBody Map<String, String> body) {
		IntentResult result = intent.predict(body.get("text"));
		return toView(result);
	}

	@Operation(summary = "批量意图预测", description = "传入 texts 数组，返回每条文本的 Top-1 意图与置信度列表")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "预测成功",
			content = @Content(mediaType = "application/json",
				schema = @Schema(implementation = List.class))),
		@ApiResponse(responseCode = "400", description = "请求体缺少 texts 字段", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping("/predict-batch")
	public List<Map<String, Object>> predictBatch(
		@Parameter(description = "请求体，字段: texts (中文文本数组)", required = true)
		@RequestBody Map<String, List<String>> body) {
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