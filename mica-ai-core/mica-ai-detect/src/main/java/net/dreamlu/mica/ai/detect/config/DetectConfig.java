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
package net.dreamlu.mica.ai.detect.config;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;

/**
 * 目标检测配置（模型族、路径、输入边长、阈值、类别名）。
 *
 * <p>默认值对齐 NanoDet-Plus（416×416、score 0.35、NMS 0.6）；
 * 切到 {@link DetectModelFamily#YOLOX_S} 时需同步
 * {@code family=YOLOX_S} + {@code inputSize=640}（YOLOX 官方 conf 0.5 / NMS 0.5）。
 *
 * <pre>{@code
 * // NanoDet-Plus（默认模型族）
 * DetectConfig nano = DetectConfig.builder()
 *     .modelPath("model-tools/detect/models/object_detection_nanodet_2022nov.onnx")
 *     .build();
 *
 * // YOLOX-s（精度更高，速度更慢）
 * DetectConfig yolox = DetectConfig.builder()
 *     .family(DetectModelFamily.YOLOX_S)
 *     .modelPath("model-tools/detect/models/object_detection_yolox_2022nov.onnx")
 *     .inputSize(640)
 *     .scoreThreshold(0.5f)
 *     .nmsThreshold(0.5f)
 *     .build();
 * }</pre>
 */
@Getter
@Builder(toBuilder = true)
@AllArgsConstructor
public class DetectConfig {

	/**
	 * 模型 ONNX 路径（必填，支持 {@code classpath:} 前缀）。
	 *
	 * <p>模型随仓库分发但不打包进 jar，必须显式指向本地文件，例如
	 * {@code model-tools/detect/models/object_detection_nanodet_2022nov.onnx}。
	 */
	private String modelPath;

	/**
	 * 模型族，决定预处理 / 解码 / NMS 契约
	 */
	@Builder.Default
	private DetectModelFamily family = DetectModelFamily.NANODET_PLUS;

	/**
	 * 模型输入边长（方形）；与模型张量形状不一致时构造期快速失败
	 */
	@Builder.Default
	private int inputSize = 416;

	/**
	 * 置信度阈值，低于该值的候选框直接丢弃
	 */
	@Builder.Default
	private float scoreThreshold = 0.35f;

	/**
	 * NMS 的 IoU 阈值，IoU 大于该值的重叠框被抑制
	 */
	@Builder.Default
	private float nmsThreshold = 0.6f;

	/**
	 * 单帧最多输出的检测框数量（按得分降序截断）
	 */
	@Builder.Default
	private int maxDetections = 100;

	/**
	 * 类别名（索引对应模型输出的 classId）；默认 COCO 80 类
	 */
	@Builder.Default
	private String[] labels = null;

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
		if (family == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "family must not be null");
		}
		if (modelPath == null || modelPath.trim().isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"modelPath 不能为空：检测模型不随 jar 分发，"
					+ "请指向本地模型文件（如 model-tools/detect/models/"
					+ family.getModelFile() + "）");
		}
		if (inputSize <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "inputSize 必须为正数");
		}
		if (inputSize % 32 != 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"inputSize 必须为 32 的整数倍，当前为 " + inputSize);
		}
		if (scoreThreshold < 0f || scoreThreshold > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"scoreThreshold 必须在 [0,1]");
		}
		if (nmsThreshold < 0f || nmsThreshold > 1f) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"nmsThreshold 必须在 [0,1]");
		}
		if (maxDetections <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "maxDetections 必须为正数");
		}
		if (labels != null && labels.length == 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"labels 为空数组时请改为 null（表示使用默认 COCO 80 类）");
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
