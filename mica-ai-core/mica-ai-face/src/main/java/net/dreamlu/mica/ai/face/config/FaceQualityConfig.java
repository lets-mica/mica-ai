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
package net.dreamlu.mica.ai.face.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * eDifFIQA(T) 人脸质量评估配置。
 *
 * <p>输入 {@code [1, 3, 112, 112]} float32，预处理：BGR→RGB → resize 112×112 →
 *     {@code (x / 255 - 0.5) / 0.5}（即 mean=0.5, std=0.5，RGB 顺序）→ NCHW。
 *
 * <p>输出：一个 float32 标量 quality score（越高越好）。
 *
 * <p>模型 License：CC-BY-4.0（署名即可商用，详见 model-tools/face/README.md）。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class FaceQualityConfig {

	/**
	 * 模型 ONNX 路径（必填，支持 {@code classpath:} 前缀）。
	 *
	 * <p>模型随仓库分发但不打包进 jar，必须显式指向本地文件，例如
	 * {@code model-tools/face/models/ediffiqa_tiny_jun2024.onnx}。
	 */
	private String modelPath;

	/**
	 * 模型标识，仅用于日志展示，不参与路径解析
	 */
	@Builder.Default
	private String modelVersion = "ediffiqa_tiny_jun2024";

	@Builder.Default
	private int inputSize = 112;

	@Builder.Default
	private float[] mean = new float[]{0.5f, 0.5f, 0.5f};

	@Builder.Default
	private float[] std = new float[]{0.5f, 0.5f, 0.5f};

	@Builder.Default
	private float threshold = 0.5f;

	@Builder.Default
	private OrtSessionOptions onnx = OrtSessionOptions.defaults();

	public void validate() {
		if (modelPath == null || modelPath.trim().isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"modelPath 不能为空：人脸质量评估模型不随 jar 分发，"
					+ "请指向本地模型文件（如 model-tools/face/models/ediffiqa_tiny_jun2024.onnx）");
		}
		if (modelVersion == null || modelVersion.isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "modelVersion must not be null or empty");
		}
		if (inputSize <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "inputSize 必须为正数");
		}
		if (mean == null || mean.length != 3 || std == null || std.length != 3) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "mean / std 必须为长度 3 的数组");
		}
		for (int i = 0; i < 3; i++) {
			if (std[i] == 0f) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "std[" + i + "] 不能为 0");
			}
		}
		if (threshold < 0f || threshold > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "threshold 必须在 [0,1]");
		}
	}

	/**
	 * 解析实际使用的模型路径。
	 *
	 * @return 配置的模型路径（原样返回；{@code classpath:} 前缀由会话层解析）
	 */
	public String resolveModelPath() {
		return modelPath;
	}
}
