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
package net.dreamlu.mica.ai.face.autoconfigure;

import lombok.Getter;
import lombok.Setter;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.NestedConfigurationProperty;

/**
 * mica-ai-face 配置属性，对应 {@code mica.ai.face} 前缀。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.ai.face")
public class FaceProperties {

	/**
	 * 模型路径配置（检测 / 识别 / 活体，均未配置时启动失败）。
	 */
	private Model model = new Model();

	/**
	 * 人脸检测参数配置。
	 */
	private Detection detection = new Detection();

	/**
	 * 活体检测参数配置。
	 */
	private Liveness liveness = new Liveness();

	/**
	 * 1:1 比对参数配置。
	 */
	private Verify verify = new Verify();

	/**
	 * 头像提取参数配置。
	 */
	private Avatar avatar = new Avatar();

	/**
	 * 证件卡片提取参数配置。
	 */
	private Card card = new Card();

	/**
	 * ONNX Runtime 会话参数（线程数 / 设备等）。
	 */
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

	/**
	 * 模型路径配置。
	 */
	@Getter
	@Setter
	public static class Model {
		/**
		 * 人脸检测（YuNet）模型路径。
		 */
		private ModelEntry detection = new ModelEntry();
		/**
		 * 人脸识别（SFace）模型路径。
		 */
		private ModelEntry recognition = new ModelEntry();
		/**
		 * 活体检测（MiniFASNetV2）模型路径。
		 */
		private ModelEntry liveness = new ModelEntry();
	}

	/**
	 * 单个模型的路径配置。
	 */
	@Getter
	@Setter
	public static class ModelEntry {
		/**
		 * 模型 ONNX 路径（支持 classpath: 前缀）。
		 */
		private String path;
	}

	/**
	 * 人脸检测参数配置。
	 */
	@Getter
	@Setter
	public static class Detection {
		/**
		 * 检测置信度阈值（0~1）。
		 */
		private float threshold = 0.9f;
		/**
		 * 检测 NMS IoU 阈值。
		 */
		private float nmsThreshold = 0.3f;
	}

	/**
	 * 活体检测参数配置。
	 */
	@Getter
	@Setter
	public static class Liveness {
		/**
		 * 是否启用活体检测（关闭后 detect 结果 liveness 字段为空）。
		 */
		private boolean enabled = true;
		/**
		 * 活体判定阈值（0~1，score 低于该值判定为攻击）。
		 */
		private float threshold = 0.85f;
		/**
		 * 人脸框外扩比例（从检测框放大到活体推理的裁剪区域）。
		 */
		private double cropScale = 2.7;
	}

	/**
	 * 1:1 比对参数配置。
	 */
	@Getter
	@Setter
	public static class Verify {
		/**
		 * 余弦相似度阈值（0~1，达到该值判定为同一人）。
		 */
		private float threshold = 0.35f;
	}

	/**
	 * 头像提取参数配置。
	 */
	@Getter
	@Setter
	public static class Avatar {
		/**
		 * 输出头像边长（正方形，像素）。
		 */
		private int size = 256;
		/**
		 * 人脸框外扩比例（相对检测框放大裁剪）。
		 */
		private double faceScale = 1.6;
		/**
		 * 头像垂直偏移（正值上移，单位为归一化比例）。
		 */
		private double verticalOffset = 0.0;
		/**
		 * 是否按关键点旋转摆正头像。
		 */
		private boolean deRotate = true;
		/**
		 * 额外旋转角度（度，配合 deRotate 使用）。
		 */
		private double rotationDegrees = 0.0;
		/**
		 * 是否按关键点方位自动校正（左右镜像判断）。
		 */
		private boolean autoOrient = true;
		/**
		 * 旋转露出的背景填充色（十六进制，如 #FFFFFF）。
		 */
		private String background = "#FFFFFF";
		/**
		 * 大图是否启用分块检测兜底（整图检测不到人脸时按块再检）。
		 */
		private boolean tileDetect = true;
		/**
		 * 分块检测的块边长（像素）。
		 */
		private int tileSize = 480;
		/**
		 * 分块重叠比例（0~1）。
		 */
		private double tileOverlap = 0.30;
		/**
		 * 分块检测置信度阈值（0~1，一般高于整图阈值以减少误检）。
		 */
		private double tileThreshold = 0.60;
		/**
		 * 最小人脸边长（像素，小于该值的人脸忽略）。
		 */
		private int minFaceSize = 40;
		/**
		 * 最多提取头像数（0 表示不限制）。
		 */
		private int maxFaces = 0;
	}

	/**
	 * 证件卡片提取参数配置。
	 */
	@Getter
	@Setter
	public static class Card {
		/**
		 * 矫正后卡片输出宽度（像素，默认 1011 对应身份证比例）。
		 */
		private int outputWidth = 1011;
		/**
		 * 矫正后卡片输出高度（像素，默认 638）。
		 */
		private int outputHeight = 638;
		/**
		 * 四边形长宽比容差（0~1，超出比例容差的候选丢弃）。
		 */
		private double aspectTolerance = 0.35;
		/**
		 * 是否启用增强（USM 锐化 + CLAHE）。
		 */
		private boolean enhance = true;
		/**
		 * USM 锐化强度（enhance 开启时生效）。
		 */
		private double sharpenAmount = 1.5;
		/**
		 * USM 锐化 sigma（enhance 开启时生效）。
		 */
		private double sharpenSigma = 1.2;
		/**
		 * 卡片最小边长（像素，小于该值的候选丢弃）。
		 */
		private int minCardSize = 400;
	}
}
