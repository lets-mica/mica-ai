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
package net.dreamlu.mica.ai.classify.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * PULC 图像分类配置（模型路径、配套 label_list、预处理契约、输出语义）。
 *
 * <p>契约来源为 PaddleClas 官方推理配置（2026-10-30 实测）：
 * <pre>
 * Eval/Infer transforms:
 *   DecodeImage:     {to_rgb: True}
 *   ResizeImage:     {resize_short: 256}
 *   CropImage:       {size: 224}
 *   NormalizeImage:  {mean: [0.485,0.456,0.406], std: [0.229,0.224,0.225], scale: 1/255}
 *   ToCHWImage
 * </pre>
 * 即「短边缩放到 256 → 中心裁剪 224×224 → RGB 归一化」，<b>不是</b>直接 resize 到 224。
 * 例外：{@code table_attribute} 的官方配置直接 {@code ResizeImage: {size: [224,224]}}，
 * 把 {@link #resizeShort} 置 {@code 0} 即可关闭「短边缩放 + 中心裁剪」。
 *
 * <p><b>模型可插拔</b>：PULC 是共用 PP-LCNet backbone 的分类家族，切换同族模型只需改
 * {@link #modelPath} 与 {@link #labelsPath}（两者必须成对配置），必要时调整
 * {@link #multiLabel} / {@link #resizeShort} / {@link #topK}。
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class ClassifyConfig {

	/**
	 * 内置 classpath 资源模板；{@code %s} 为 {@link #modelVersion}
	 */
	private static final String MODEL_RESOURCE = "mica-ai/models/classify/%s.onnx";

	/**
	 * 自定义模型路径；为空时按 {@code modelVersion} 解析内置 classpath 路径
	 */
	private String modelPath;

	/**
	 * 模型标识（对应 {@code model-tools/classify/models/} 下的文件名前缀），
	 * 默认首批的 {@code text_image_orientation}
	 */
	@Builder.Default
	private String modelVersion = "text_image_orientation";

	/**
	 * 与模型强耦合的 label_list 文件路径（必填，无 classpath 兜底）。
	 *
	 * <p>格式为「一行一个标签」，按行序对应模型输出索引；<b>刻意不做去重</b>
	 * （去重会让索引整体错位）。类别数与资源条目数不一致时快速失败。
	 */
	private String labelsPath;

	/**
	 * 模型输入宽度（NCHW 最后一维）；官方 PULC 导出为 224
	 */
	@Builder.Default
	private int inputWidth = 224;

	/**
	 * 模型输入高度（NCHW 倒数第二维）；官方 PULC 导出为 224
	 */
	@Builder.Default
	private int inputHeight = 224;

	/**
	 * 短边缩放目标；大于 0 时执行「短边缩放到该值 + 中心裁剪 inputWidth×inputHeight」，
	 * 官方 PULC 默认 256；置 0 时直接 resize 到 inputWidth×inputHeight
	 * （{@code table_attribute} 这类官方直接 resize 的模型用 0）。
	 */
	@Builder.Default
	private int resizeShort = 256;

	/**
	 * 输入通道顺序；默认 {@link ClassifyChannelOrder#RGB}（PaddleClas 官方约定）
	 */
	@Builder.Default
	private ClassifyChannelOrder channelOrder = ClassifyChannelOrder.RGB;

	/**
	 * 归一化均值（RGB 3 通道，PaddleClas {@code NormalizeImage} 默认值）
	 */
	@Builder.Default
	private float[] mean = new float[]{0.485f, 0.456f, 0.406f};

	/**
	 * 归一化标准差（RGB 3 通道，PaddleClas {@code NormalizeImage} 默认值）
	 */
	@Builder.Default
	private float[] std = new float[]{0.229f, 0.224f, 0.225f};

	/**
	 * 缩放插值方式，默认双线性
	 */
	@Builder.Default
	private ClassifyInterpolation interpolation = ClassifyInterpolation.LINEAR;

	/**
	 * 是否多标签（每个输出独立二分类）。
	 *
	 * <p>{@code false}：单标签多分类，输出按 softmax 归一化，{@code topK} 取最大若干类；
	 * {@code true}：多标签，每个输出独立 sigmoid，各属性互不影响
	 * （如 {@code table_attribute} 的 6 个表格属性）。
	 */
	@Builder.Default
	private boolean multiLabel = false;

	/**
	 * 模型输出是否已是概率（softmax / sigmoid 后）。
	 *
	 * <p>实测 PULC 官方导出已内置 softmax（输出节点 {@code softmax_1.tmp_0}）或
	 * sigmoid（{@code sigmoid_2.tmp_0}），故默认 {@code true}；
	 * 若换到输出裸 logits 的导出，置 {@code false} 由本模块补做激活。
	 */
	@Builder.Default
	private boolean outputIsProbability = true;

	/**
	 * 返回前 K 个类别（仅单标签模式有意义；多标签模式最多返回全部类别）
	 */
	@Builder.Default
	private int topK = 1;

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）
	 */
	@Builder.Default
	private OrtSessionOptions onnx = OrtSessionOptions.defaults();

	/**
	 * 校验配置合法性，非法时抛出 {@link MicaAiException}。
	 *
	 * @throws MicaAiException 任一配置项非法时抛出
	 */
	public void validate() {
		if (modelVersion == null || modelVersion.isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"modelVersion must not be null or empty");
		}
		if (labelsPath == null || labelsPath.isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"labelsPath 必填：分类模型必须与 label_list 成对配置");
		}
		if (inputWidth <= 0 || inputHeight <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"inputWidth / inputHeight 必须为正数");
		}
		if (resizeShort < 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"resizeShort 不能为负数（0 表示关闭短边缩放 + 中心裁剪）");
		}
		if (resizeShort > 0 && resizeShort < Math.max(inputWidth, inputHeight)) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"resizeShort(" + resizeShort + ") 必须不小于输入尺寸 ("
					+ inputWidth + "x" + inputHeight + ")，否则中心裁剪会越界");
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
		if (channelOrder == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"channelOrder must not be null");
		}
		if (interpolation == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"interpolation must not be null");
		}
		if (topK <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"topK 必须为正数");
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
