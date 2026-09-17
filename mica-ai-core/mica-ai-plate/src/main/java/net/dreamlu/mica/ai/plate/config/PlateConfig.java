/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.plate.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PlateConfig {

    private String detectionModelPath;
    private String recognitionModelPath;
    private String classificationModelPath;

    @Builder.Default
    private String modelVersion = "20230229";

    @Builder.Default
    private int detectionInputSize = 320;

    @Builder.Default
    private int recognitionInputHeight = 48;

    @Builder.Default
    private int recognitionInputWidth = 160;

    @Builder.Default
    private int classificationInputSize = 96;

    @Builder.Default
    private float detectionConfidenceThreshold = 0.25f;

    @Builder.Default
    private float detectionNmsThreshold = 0.5f;

    @Builder.Default
    private int maxPlates = 5;

    @Builder.Default
	private OrtSessionOptions onnx = OrtSessionOptions.defaults();

    public void validate() {
        if (modelVersion == null || modelVersion.isEmpty()) {
            throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
                "modelVersion must not be null or empty");
        }
        if (detectionInputSize != 320 && detectionInputSize != 640) {
            throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
                "detectionInputSize 仅支持 320 或 640");
        }
        if (maxPlates <= 0) {
            throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
                "maxPlates 必须为正数");
        }
    }
}
