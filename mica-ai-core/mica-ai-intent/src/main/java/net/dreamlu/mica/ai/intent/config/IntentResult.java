/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent.config;

import java.util.Map;

/**
 * 意图推理结果。
 *
 * @param intent     最佳匹配意图
 * @param confidence 置信度 [0, 1]
 * @param allScores  所有意图的分数（意图标签 → 概率）
 */
public record IntentResult(
	String intent,
	float confidence,
	Map<String, Float> allScores
) {

}
