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

	private Model model = new Model();
	private Detection detection = new Detection();
	private Liveness liveness = new Liveness();
	private Verify verify = new Verify();
	private Avatar avatar = new Avatar();
	private Card card = new Card();
	@NestedConfigurationProperty
	private OrtSessionOptions onnx = new OrtSessionOptions();

	/** 模型路径配置。 */
	@Getter
	@Setter
	public static class Model {
		private ModelEntry detection = new ModelEntry();
		private ModelEntry recognition = new ModelEntry();
		private ModelEntry liveness = new ModelEntry();
	}

	/** 单个模型的路径配置。 */
	@Getter
	@Setter
	public static class ModelEntry {
		private String path;
	}

	/** 人脸检测参数配置。 */
	@Getter
	@Setter
	public static class Detection {
		private float threshold = 0.9f;
		private float nmsThreshold = 0.3f;
	}

	/** 活体检测参数配置。 */
	@Getter
	@Setter
	public static class Liveness {
		private boolean enabled = true;
		private float threshold = 0.85f;
		private double cropScale = 2.7;
	}

	/** 1:1 比对参数配置。 */
	@Getter
	@Setter
	public static class Verify {
		private float threshold = 0.35f;
	}

	/** 头像提取参数配置。 */
	@Getter
	@Setter
	public static class Avatar {
		private int size = 256;
		private double faceScale = 1.6;
		private double verticalOffset = 0.0;
		private boolean deRotate = true;
		private double rotationDegrees = 0.0;
		private boolean autoOrient = true;
		private String background = "#FFFFFF";
		private boolean tileDetect = true;
		private int tileSize = 480;
		private double tileOverlap = 0.30;
		private double tileThreshold = 0.60;
		private int minFaceSize = 40;
		private int maxFaces = 0;
	}

	/** 证件卡片提取参数配置。 */
	@Getter
	@Setter
	public static class Card {
		private int outputWidth = 1011;
		private int outputHeight = 638;
		private double aspectTolerance = 0.35;
		private boolean enhance = true;
		private double sharpenAmount = 1.5;
		private double sharpenSigma = 1.2;
		private int minCardSize = 400;
	}
}
