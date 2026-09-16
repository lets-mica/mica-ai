/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.plate.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * mica-ai-plate 配置属性，对应 {@code mica.ai.plate} 前缀。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.plate")
public class PlateProperties {

    private boolean enabled = true;
    private String modelVersion = "20230229";
    private String detectionModelPath;
    private String recognitionModelPath;
    private String classificationModelPath;
    private int detectionInputSize = 320;
    private int recognitionInputHeight = 48;
    private int recognitionInputWidth = 160;
    private int classificationInputSize = 96;
    private float detectionConfidenceThreshold = 0.25f;
    private float detectionNmsThreshold = 0.5f;
    private int maxPlates = 5;
    private String device = "cpu";
    private Onnx onnx = new Onnx();

    @Getter
    @Setter
    public static class Onnx {
        private int intraOpNumThreads = 0;
        private int interOpNumThreads = 0;
        private int cudaDeviceId = 0;
    }
}