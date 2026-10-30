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
package net.dreamlu.mica.ai.example.config;

import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 示例应用自有的分类模型配置，对应 {@code mica.example.classify} 前缀。
 *
 * <p>{@code mica-ai-classify} starter 的一个引擎只绑定「一个模型 + 一份 label_list」；
 * 要同时用多个 PULC 模型，就得再声明引擎。这里把多余的模型声明在示例自己的前缀下，
 * 与 starter 的单模型默认装配并存，互不干扰。
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "mica.example.classify")
public class ClassifyModelProperties {

	/**
	 * 额外声明的 PULC 模型：key = 模型名，value = 模型与配套 label_list。
	 */
	private Map<String, ModelEntry> models = new LinkedHashMap<>();

	/**
	 * 单个 PULC 模型的配置。
	 */
	@Getter
	@Setter
	public static class ModelEntry {

		/**
		 * 模型 ONNX 路径。
		 */
		private String modelPath;

		/**
		 * 与模型强耦合的 label_list 路径（必填）。
		 */
		private String labelsPath;

		/**
		 * 短边缩放目标；0 表示直接 resize 到输入尺寸。
		 */
		private int resizeShort = 256;

		/**
		 * 是否多标签（各属性独立 sigmoid）。
		 */
		private boolean multiLabel = false;

		/**
		 * 返回前 K 个类别。
		 */
		private int topK = 1;
	}
}
