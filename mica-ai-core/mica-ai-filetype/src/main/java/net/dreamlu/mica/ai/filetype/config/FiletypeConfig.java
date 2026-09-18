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
package net.dreamlu.mica.ai.filetype.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * Magika 文件类型检测配置（模型 / 知识库路径、预测模式、ONNX 会话参数）。
 */
@Getter
@Builder(toBuilder = true)
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
	private OrtSessionOptions onnx = OrtSessionOptions.defaults();

	/**
	 * 校验配置合法性。
	 *
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 配置非法
	 */
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

	/**
	 * 解析模型 ONNX 文件路径（未配置时回退内置 classpath 资源）。
	 *
	 * @return 模型路径
	 */
    public String resolveModelPath() {
        return resolveResource(modelPath, MODEL_RESOURCE);
    }

	/**
	 * 解析 config.min.json 路径（未配置时回退内置 classpath 资源）。
	 *
	 * @return 配置路径
	 */
    public String resolveConfigPath() {
        return resolveResource(configPath, CONFIG_RESOURCE);
    }

	/**
	 * 解析 content_types_kb.min.json 路径（未配置时回退内置 classpath 资源）。
	 *
	 * @return 知识库路径
	 */
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
