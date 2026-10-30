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
package net.dreamlu.mica.ai.humanseg.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * 人像分割配置（模型路径、输入宽高、归一化参数、二值阈值）。
 *
 * <p><b>同族可插拔</b>：PP-HumanSeg 的契约参数全部显式化，切换同族模型只需调整
 * {@link #modelPath} 与 {@link #inputWidth} / {@link #inputHeight}：
 * <ul>
 *   <li>{@code human_segmentation_pphumanseg_2023mar}（5.88 MB，已入库）—— 192×192 方形输入</li>
 *   <li>Portrait-PP-HumanSegV2-Lite（2.3 MB，外置）—— 256×144 竖版输入（16:9，需成对改宽高）</li>
 * </ul>
 * 归一化统一为 {@code (x/255 - 0.5) / 0.5}（PaddleSeg 默认的 mean=std=0.5）。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class HumanSegConfig {

	/**
	 * 内置 classpath 资源模板；{@code %s} 为 {@link #modelVersion}
	 */
	private static final String MODEL_RESOURCE = "mica-ai/models/humanseg/%s.onnx";

	/**
	 * 自定义模型路径；为空时按 {@code modelVersion} 解析内置 classpath 路径
	 */
	private String modelPath;

	/**
	 * 模型标识（对应 {@code model-tools/humanseg/models/} 下的文件名前缀，也是内置 classpath 资源名）
	 */
	@Builder.Default
	private String modelVersion = "human_segmentation_pphumanseg_2023mar";

	/**
	 * 模型输入宽度（NCHW 最后一维）；与模型形状不一致时构造期快速失败
	 */
	@Builder.Default
	private int inputWidth = 192;

	/**
	 * 模型输入高度（NCHW 倒数第二维）；与模型形状不一致时构造期快速失败
	 */
	@Builder.Default
	private int inputHeight = 192;

	/**
	 * 归一化均值（RGB 3 通道，PaddleSeg 默认 0.5）
	 */
	@Builder.Default
	private float[] mean = new float[]{0.5f, 0.5f, 0.5f};

	/**
	 * 归一化标准差（RGB 3 通道，PaddleSeg 默认 0.5）
	 */
	@Builder.Default
	private float[] std = new float[]{0.5f, 0.5f, 0.5f};

	/**
	 * 二值掩码阈值（0~1）；仅 {@code maskBinary} 系列使用
	 */
	@Builder.Default
	private float binaryThreshold = 0.5f;

	/**
	 * 纯色底输出的默认底色（RGB，0~255）
	 */
	@Builder.Default
	private int[] backgroundColor = new int[]{255, 255, 255};

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）
	 */
	@Builder.Default
	private OrtSessionOptions onnx = OrtSessionOptions.defaults();

	/**
	 * 校验配置合法性，非法时抛出 {@link MicaAiException}。
	 *
	 * @throws MicaAiException 任一配置项非法时抛出，错误码为 {@link ErrorCode#ILLEGAL_ARGUMENT}
	 */
	public void validate() {
		if (modelVersion == null || modelVersion.isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"modelVersion must not be null or empty");
		}
		if (inputWidth <= 0 || inputHeight <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"inputWidth / inputHeight 必须为正数");
		}
		if (mean == null || mean.length != 3 || std == null || std.length != 3) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"mean / std 必须为长度 3 的数组");
		}
		for (int i = 0; i < 3; i++) {
			if (std[i] == 0f) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"std[" + i + "] 不能为 0");
			}
		}
		if (binaryThreshold < 0f || binaryThreshold > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"binaryThreshold 必须在 [0,1]");
		}
		if (backgroundColor == null || backgroundColor.length != 3) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"backgroundColor 必须为长度 3 的数组");
		}
		for (int i = 0; i < 3; i++) {
			if (backgroundColor[i] < 0 || backgroundColor[i] > 255) {
				throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
					"backgroundColor[" + i + "] 必须在 [0,255]");
			}
		}
	}

	/**
	 * 解析实际使用的模型路径。
	 *
	 * @return 配置了 {@code modelPath} 时直接返回；否则返回内置 classpath 资源路径
	 */
	public String resolveModelPath() {
		if (modelPath != null && !modelPath.isEmpty()) {
			return modelPath;
		}
		return OnnxModelSession.CLASSPATH_PREFIX
			+ String.format(MODEL_RESOURCE, modelVersion);
	}
}
