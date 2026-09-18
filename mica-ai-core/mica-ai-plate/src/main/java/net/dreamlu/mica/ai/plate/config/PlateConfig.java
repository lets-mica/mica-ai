/*
 * Copyright (c) 2019-2029, Dreamlu 卢春梦 (596392912@qq.com & dreamlu.net).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.dreamlu.mica.ai.plate.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

@Getter
@Builder(toBuilder = true)
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
