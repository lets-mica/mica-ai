/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 单条人脸特征匹配结果。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class MatchResult {

	private String personId;
	private float similarity;
}