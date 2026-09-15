/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import net.dreamlu.mica.ai.common.exception.MicaAiException;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FiletypeConfig {

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
            throw new MicaAiException(MicaAiException.ErrorCode.ILLEGAL_ARGUMENT,
                "modelVersion must not be null or empty");
        }
        if (predictionMode == null) {
            throw new MicaAiException(MicaAiException.ErrorCode.ILLEGAL_ARGUMENT,
                "predictionMode must not be null");
        }
    }

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class OnnxOptions {

        @Builder.Default
        private int intraOpNumThreads = 0;

        @Builder.Default
        private int interOpNumThreads = 0;

        @Builder.Default
        private boolean gpu = false;

        @Builder.Default
        private int cudaDeviceId = 0;

        public static OnnxOptions defaults() {
            return OnnxOptions.builder().build();
        }
    }
}
