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
package net.dreamlu.mica.ai.textline.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * PP-LCNet 文本行方向分类配置（模型路径、输入尺寸、归一化参数、通道顺序等）。
 *
 * <p>契约来源为官方推理包自带的 {@code inference.yml}（实测 2026-09-21）：
 * <pre>
 * PreProcess:
 *   transform_ops:
 *   - ResizeImage:   {size: [160, 80]}          # (宽, 高)
 *   - NormalizeImage: {mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225], scale: 1/255}
 *   - ToCHWImage
 * PostProcess:
 *   Topk: {topk: 1, label_list: [0_degree, 180_degree]}
 * </pre>
 *
 * <p><b>模型可插拔</b>：{@code modelVersion} 与 {@link #modelPath} 决定加载哪个模型，
 * 本模块内置 {@code PP-LCNet_x1_0_textline_ori}；同任务的
 * {@code PP-LCNet_x0_25_textline_ori}（0.96MB，更轻）契约一致，改路径即可切换。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class TextLineConfig {

	/**
	 * 内置 classpath 资源模板；{@code %s} 为模型文件名（不含扩展名）
	 */
	private static final String MODEL_RESOURCE = "mica-ai/models/textline/%s.onnx";

	/**
	 * 自定义模型路径；为空时按 {@code modelVersion} 解析内置 classpath 路径
	 */
	private String modelPath;

	/**
	 * 模型标识（对应 {@code model-tools/textline/models/} 下的文件名前缀，
	 * 也是内置 classpath 资源名）；本模块内置
	 * {@code PP-LCNet_x1_0_textline_ori}，轻量版 {@code PP-LCNet_x0_25_textline_ori}
	 * 通过 {@link #modelPath} 外置接入
	 */
	@Builder.Default
	private String modelVersion = "PP-LCNet_x1_0_textline_ori";

	/**
	 * 模型输入宽度（W，NCHW 最后一维），官方导出为 160；与模型不一致时构造期快速失败
	 */
	@Builder.Default
	private int inputWidth = 160;

	/**
	 * 模型输入高度（H，NCHW 倒数第二维），官方导出为 80；与模型不一致时构造期快速失败
	 */
	@Builder.Default
	private int inputHeight = 80;

	/**
	 * 输入通道顺序；默认 {@link TextLineChannelOrder#BGR}（官方 PaddleX 约定）。
	 *
	 * <p>本任务实测对通道顺序不敏感，该配置项面向换模型场景，见枚举说明。
	 */
	@Builder.Default
	private TextLineChannelOrder channelOrder = TextLineChannelOrder.BGR;

	/**
	 * 归一化均值（3 通道，PaddleX {@code NormalizeImage} 默认值）。
	 *
	 * <p>注意：PaddleX 按 BGR 顺序逐通道施加，故默认值在此语境下即 BGR 顺序的
	 * {@code [0.485, 0.456, 0.406]}。
	 */
	@Builder.Default
	private float[] mean = new float[]{0.485f, 0.456f, 0.406f};

	/**
	 * 归一化标准差（3 通道，PaddleX {@code NormalizeImage} 默认值）
	 */
	@Builder.Default
	private float[] std = new float[]{0.229f, 0.224f, 0.225f};

	/**
	 * 缩放进模型尺寸时使用的插值方式（OpenCV 常量由内部映射），默认双线性
	 */
	@Builder.Default
	private TextLineInterpolation interpolation = TextLineInterpolation.LINEAR;

	/**
	 * 判定为「倒置」的最小概率；低于该值时仍按 {@code 0 度} 处理。
	 *
	 * <p>本模型 2 类 softmax 通常给出较极端的概率；设为 {@code 0.5} 即纯 argmax。
	 * 调高可让「拿不准」的行保持不动，避免误旋转反而破坏原本正确的行。
	 */
	@Builder.Default
	private float upsideDownThreshold = 0.5f;

	/**
	 * 模型输出是否已是概率（softmax 后）。
	 *
	 * <p>实测 {@code PP-LCNet_x1_0_textline_ori} 的 ONNX 输出为 <b>原始 logits</b>
	 * （如 {@code [+1.0000, +0.0000]}），需要 softmax 才能当概率用，故默认 {@code false}。
	 */
	@Builder.Default
	private boolean outputIsProbability = false;

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
		if (inputWidth <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"inputWidth 必须为正数");
		}
		if (inputHeight <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"inputHeight 必须为正数");
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
		if (upsideDownThreshold < 0f || upsideDownThreshold > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"upsideDownThreshold 必须在 [0,1]");
		}
		if (channelOrder == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"channelOrder must not be null");
		}
		if (interpolation == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"interpolation must not be null");
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
