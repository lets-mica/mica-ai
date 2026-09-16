/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.plate.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class PlateResult {

    private String plateCode;
    private PlateType plateType;
    private float detectionConfidence;
    private float recognitionConfidence;
    private int[] boundingBox;
    private int[][] landmarks;
}