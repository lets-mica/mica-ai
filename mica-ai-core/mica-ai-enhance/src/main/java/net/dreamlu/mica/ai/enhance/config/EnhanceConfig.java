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
package net.dreamlu.mica.ai.enhance.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * 图像超分配置（模型路径、放大倍数、tile 参数）。
 *
 * <p>当前入库模型为 {@code realesr-general-x4v3}（2.33MB，动态 H/W 输入，4 倍放大）。
 * 同族可插拔（契约一致，只改 {@code modelPath} + {@code scale}）：
 * <ul>
 *   <li>{@code realesr-general-x4v3}（2.33MB，入库）—— 默认</li>
 *   <li>{@code realesr-general-wdn-x4v3}（去噪版，外置）—— 需自行转换 ONNX</li>
 * </ul>
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class EnhanceConfig {

	/**
	 * 模型 ONNX 路径（必填，支持 {@code classpath:} 前缀）。
	 *
	 * <p>模型随仓库分发但不打包进 jar，必须显式指向本地文件，例如
	 * {@code model-tools/enhance/models/realesr-general-x4v3.onnx}；
	 * 同族的 wdn 去噪版同样通过本字段指向本地文件接入。
	 */
	private String modelPath;

	/**
	 * 模型标识，仅用于日志展示，不参与路径解析
	 */
	@Builder.Default
	private String modelVersion = "realesr-general-x4v3";

	/**
	 * 放大倍数；与模型实际输出/输入比值不一致时推理期快速失败
	 */
	@Builder.Default
	private int scale = 4;

	/**
	 * 单次推理的 tile 边长（原图像素）；图像小于该值时整图推理
	 */
	@Builder.Default
	private int tileSize = 256;

	/**
	 * tile 重叠边宽（原图像素），用于消除拼接接缝；必须小于 {@link #tileSize}
	 */
	@Builder.Default
	private int tileOverlap = 16;

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
				"modelPath 不能为空：图像超分模型不随 jar 分发，"
					+ "请指向本地模型文件（如 model-tools/enhance/models/realesr-general-x4v3.onnx）");
		}
		if (modelVersion == null || modelVersion.isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"modelVersion must not be null or empty");
		}
		if (scale <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "scale 必须为正数");
		}
		if (tileSize <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "tileSize 必须为正数");
		}
		if (tileOverlap < 0 || tileOverlap >= tileSize) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"tileOverlap 必须在 [0, tileSize) 区间内");
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
