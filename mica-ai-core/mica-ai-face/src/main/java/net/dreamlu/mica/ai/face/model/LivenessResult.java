/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 活体检测结果。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class LivenessResult {

	private float liveScore;
	private boolean isLive;
	private String attackType;
}