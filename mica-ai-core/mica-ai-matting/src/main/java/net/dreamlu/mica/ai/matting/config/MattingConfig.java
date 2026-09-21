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
package net.dreamlu.mica.ai.matting.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * U-2-Net / u2netp 抠图配置（模型路径、输入边长、归一化参数、输出节点等）。
 *
 * <p><b>模型可插拔</b>：本配置把「U²-Net 族」的契约参数全部显式化，因此
 * <b>rembg 导出的同族模型无需改 Java 代码即可切换</b>，只需调整
 * {@link #modelPath}（模型文件）与个别契约参数：
 * <ul>
 *   <li>{@code u2netp}（4.36 MB，已入库）—— 轻量，CPU 毫秒级</li>
 *   <li>{@code u2net}（168 MB，外置）—— 通用显著性完整版，边缘更细</li>
 *   <li>{@code u2net_human_seg}（168 MB，外置）—— 人体分割专用，对「无人图」更保守</li>
 * </ul>
 * 实测（2026-09-21）三者契约完全一致：{@code input.1 [1,3,320,320]} 输入、
 * 7 个 {@code [1,1,320,320]} 输出、d0 为首个输出、ImageNet RGB 归一化。
 * 超限模型（&gt;50MB）按 AGENTS.md §6.2 不入库，指向本地路径即可。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class MattingConfig {

	/**
	 * 内置 classpath 资源模板；{@code %s} 为模型文件名（不含扩展名）
	 */
	private static final String MODEL_RESOURCE = "mica-ai/models/matting/%s.onnx";

	/**
	 * 自定义模型路径；为空时按 {@code modelVersion} 解析内置 classpath 路径
	 */
	private String modelPath;

	/**
	 * 模型标识（对应 {@code model-tools/matting/models/} 下的文件名前缀，
	 * 也是内置 classpath 资源名）；本模块内置 {@code u2netp} 一个模型，
	 * 其它同族模型通过 {@link #modelPath} 外置接入
	 */
	@Builder.Default
	private String modelVersion = "u2netp";

	/**
	 * 模型输入边长，U²-Net 族官方导出为 320；与模型不一致时构造期快速失败
	 */
	@Builder.Default
	private int inputSize = 320;

	/**
	 * 输出节点选择策略，决定从 7 个输出里怎么定位 d0；默认 {@link MattingOutputSelect#AUTO}
	 */
	@Builder.Default
	private MattingOutputSelect outputSelect = MattingOutputSelect.DEFAULT;

	/**
	 * 归一化均值（RGB 3 通道，ImageNet 统计量，U²-Net 族官方训练配置）
	 */
	@Builder.Default
	private float[] mean = new float[]{0.485f, 0.456f, 0.406f};

	/**
	 * 归一化标准差（RGB 3 通道，ImageNet 统计量，U²-Net 族官方训练配置）
	 */
	@Builder.Default
	private float[] std = new float[]{0.229f, 0.224f, 0.225f};

	/**
	 * 缩放掩码时使用的插值方式；默认线性插值，避免 alpha 边缘出现锯齿
	 */
	@Builder.Default
	private MattingInterpolation interpolation = MattingInterpolation.LINEAR;

	/**
	 * 二值掩码阈值（0~1）；仅 {@code matteBinary} 系列使用
	 */
	@Builder.Default
	private float binaryThreshold = 0.5f;

	/**
	 * 是否把 d0 做 min-max 拉伸到 [0,1]。
	 *
	 * <p>U²-Net 族的 d0 已经是 Sigmoid 输出（实测落在 [0,1]），但 min-max 拉伸
	 * 是 rembg 的参考行为：它把「该图内的相对显著性」铺满整个 alpha 值域，
	 * 对低对比度主体能显著提升可见度（实测放大 ~470×）。关闭后直接使用模型原始概率。
	 */
	@Builder.Default
	private boolean minMaxNormalize = true;

	/**
	 * 抠图结果默认输出底色（RGB，0~255），{@code transparent=true} 时忽略
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
		if (inputSize <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"inputSize 必须为正数");
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
		if (interpolation == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"interpolation must not be null");
		}
		if (outputSelect == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"outputSelect must not be null");
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
