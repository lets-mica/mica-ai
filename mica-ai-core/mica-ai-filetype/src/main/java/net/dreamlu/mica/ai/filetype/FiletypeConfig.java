/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OnnxOptions;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FiletypeConfig {

    private static final String MODEL_RESOURCE = "mica-ai/models/filetype/%s/model.onnx";
    private static final String CONFIG_RESOURCE = "mica-ai/models/filetype/%s/config.min.json";
    private static final String KB_RESOURCE = "mica-ai/models/filetype/%s/content_types_kb.min.json";

    private String modelPath;
    private String configPath;
    private String contentTypesPath;

    @Builder.Default
    private String modelVersion = "standard_v3_3";

    @Builder.Default
    private PredictionMode predictionMode = PredictionMode.HIGH_CONFIDENCE;

    @Builder.Default
    private OnnxOptions onnx = OnnxOptions.defaults();

    public void validate() {
        if (modelVersion == null || modelVersion.isEmpty()) {
            throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
                "modelVersion must not be null or empty");
        }
        if (predictionMode == null) {
            throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
                "predictionMode must not be null");
        }
    }

    public String resolveModelPath() {
        return resolveResource(modelPath, MODEL_RESOURCE);
    }

    public String resolveConfigPath() {
        return resolveResource(configPath, CONFIG_RESOURCE);
    }

    public String resolveContentTypesPath() {
        return resolveResource(contentTypesPath, KB_RESOURCE);
    }

    private String resolveResource(String explicit, String resourceTemplate) {
        if (explicit != null && !explicit.isEmpty()) {
            return explicit;
        }
        return OnnxModelSession.CLASSPATH_PREFIX + String.format(resourceTemplate, modelVersion);
    }
}
