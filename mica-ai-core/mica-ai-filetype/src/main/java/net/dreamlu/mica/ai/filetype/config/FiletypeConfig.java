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
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * Magika 文件类型检测配置（模型 / 知识库路径、预测模式、ONNX 会话参数）。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class FiletypeConfig {

    /**
     * 模型 ONNX 路径（必填，支持 {@code classpath:} 前缀）。
     *
     * <p>模型随仓库分发但不打包进 jar，必须显式指向本地文件，例如
     * {@code model-tools/filetype/models/model.onnx}。
     */
    private String modelPath;

    /**
     * config.min.json 路径（必填，支持 {@code classpath:} 前缀）。
     *
     * <p>与模型成对使用，指向仓库内文件，例如
     * {@code model-tools/filetype/models/config.min.json}。
     */
    private String configPath;

    /**
     * content_types_kb.min.json 知识库路径（必填，支持 {@code classpath:} 前缀）。
     *
     * <p>与模型成对使用，指向仓库内文件，例如
     * {@code model-tools/filetype/models/content_types_kb.min.json}。
     */
    private String contentTypesPath;

    /**
     * 模型标识，仅用于日志展示，不参与路径解析
     */
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
        if (modelPath == null || modelPath.trim().isEmpty()) {
            throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
                "modelPath 不能为空：文件类型识别模型不随 jar 分发，"
                    + "请指向本地模型文件（如 model-tools/filetype/models/model.onnx）");
        }
        if (configPath == null || configPath.trim().isEmpty()) {
            throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
                "configPath 不能为空：Magika 配置需与模型成对使用，"
                    + "请指向本地文件（如 model-tools/filetype/models/config.min.json）");
        }
        if (contentTypesPath == null || contentTypesPath.trim().isEmpty()) {
            throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
                "contentTypesPath 不能为空：Magika 知识库需与模型成对使用，"
                    + "请指向本地文件（如 model-tools/filetype/models/content_types_kb.min.json）");
        }
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
	 * 解析模型 ONNX 文件路径。
	 *
	 * @return 配置的模型路径（原样返回；{@code classpath:} 前缀由会话层解析）
	 */
    public String resolveModelPath() {
        return modelPath;
    }

	/**
	 * 解析 config.min.json 路径。
	 *
	 * @return 配置的路径（原样返回；{@code classpath:} 前缀由加载层解析）
	 */
    public String resolveConfigPath() {
        return configPath;
    }

	/**
	 * 解析 content_types_kb.min.json 路径。
	 *
	 * @return 配置的路径（原样返回；{@code classpath:} 前缀由加载层解析）
	 */
    public String resolveContentTypesPath() {
        return contentTypesPath;
    }
}
