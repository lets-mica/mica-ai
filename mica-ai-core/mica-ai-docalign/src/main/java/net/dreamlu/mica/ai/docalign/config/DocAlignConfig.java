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
package net.dreamlu.mica.ai.docalign.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * 文档角点检测配置（模型路径、输入宽高、热力图阈值）。
 *
 * <p>当前入库模型为 {@code lcnet100_h_e_bifpn_256_fp32}（4.5MB，256×256 输入，
 * 输出 128×128 的 4 通道热力图）。同族可插拔（契约一致，只改 {@code modelPath}）：
 * <ul>
 *   <li>{@code lcnet100_h_e_bifpn_256_fp32}（4.5MB，入库）—— 默认</li>
 *   <li>{@code lcnet050_p_multi_decoder_l3_d64_256_fp32}（4.9MB，外置）</li>
 *   <li>{@code fastvit_t8_h_e_bifpn_256_fp32}（13.2MB，外置）</li>
 *   <li>{@code fastvit_sa24_h_e_bifpn_256_fp32}（83MB，外置，精度最高）</li>
 * </ul>
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class DocAlignConfig {

	/**
	 * 模型 ONNX 路径（必填，支持 {@code classpath:} 前缀）。
	 *
	 * <p>模型随仓库分发但不打包进 jar，必须显式指向本地文件，例如
	 * {@code model-tools/docalign/models/lcnet100_h_e_bifpn_256_fp32.onnx}；
	 * 同族的 fastvit 等更大模型同样通过本字段指向本地文件接入。
	 */
	private String modelPath;

	/**
	 * 模型标识，仅用于日志展示，不参与路径解析
	 */
	@Builder.Default
	private String modelVersion = "lcnet100_h_e_bifpn_256_fp32";

	/**
	 * 模型输入宽度；与模型形状不一致时构造期快速失败
	 */
	@Builder.Default
	private int inputWidth = 256;

	/**
	 * 模型输入高度；与模型形状不一致时构造期快速失败
	 */
	@Builder.Default
	private int inputHeight = 256;

	/**
	 * 热力图二值化阈值（上游默认 0.3）；低于该值的像素视为背景
	 */
	@Builder.Default
	private float heatmapThreshold = 0.3f;

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
		if (modelPath == null || modelPath.trim().isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"modelPath 不能为空：文档角点模型不随 jar 分发，"
					+ "请指向本地模型文件（如 model-tools/docalign/models/"
					+ "lcnet100_h_e_bifpn_256_fp32.onnx）");
		}
		if (modelVersion == null || modelVersion.isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"modelVersion must not be null or empty");
		}
		if (inputWidth <= 0 || inputHeight <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"inputWidth / inputHeight 必须为正数");
		}
		if (heatmapThreshold < 0f || heatmapThreshold > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"heatmapThreshold 必须在 [0,1]");
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
