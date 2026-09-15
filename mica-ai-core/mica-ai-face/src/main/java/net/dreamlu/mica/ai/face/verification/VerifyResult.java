/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.verification;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 人脸 1:1 比对结果。
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class VerifyResult {

	private float similarity;
	private float threshold;
	private boolean passed;
}