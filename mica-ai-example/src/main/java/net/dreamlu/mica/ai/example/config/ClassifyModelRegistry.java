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

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.classify.config.ClassifyConfig;
import net.dreamlu.mica.ai.classify.core.ClassifyEngine;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * PULC 多模型注册表。
 *
 * <p>starter 提供的 {@link ClassifyEngine} 是「单模型」默认装配；这里按需（首次调用时）
 * 懒加载 {@code mica.example.classify.models} 下声明的其它 PULC 模型，
 * 演示「一个引擎绑定一个模型 + 一份 label_list」的多模型用法。
 *
 * <p>模型与 label_list 严格成对：缺任一项在创建引擎时直接抛
 * {@link MicaAiException}（fail-fast），不会输出乱码。
 */
@Slf4j
@Component
@EnableConfigurationProperties(ClassifyModelProperties.class)
public class ClassifyModelRegistry implements DisposableBean {

	private final ClassifyModelProperties properties;
	private final Map<String, ClassifyEngine> engines = new ConcurrentHashMap<>();

	public ClassifyModelRegistry(ClassifyModelProperties properties) {
		this.properties = properties;
	}

	/**
	 * 已声明的额外模型名。
	 *
	 * @return 模型名集合
	 */
	public Set<String> modelNames() {
		return properties.getModels().keySet();
	}

	/**
	 * 取（并在首次调用时创建）指定模型的引擎。
	 *
	 * @param name 模型名，对应 {@code mica.example.classify.models.<name>}
	 * @return 分类引擎
	 * @throws MicaAiException 模型未声明或配置非法时抛出
	 */
	public ClassifyEngine get(String name) {
		ClassifyEngine engine = engines.get(name);
		if (engine != null) {
			return engine;
		}
		ClassifyModelProperties.ModelEntry entry = properties.getModels().get(name);
		if (entry == null) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"未声明的分类模型: " + name + "，可选: " + modelNames());
		}
		synchronized (this) {
			engine = engines.get(name);
			if (engine == null) {
				engine = ClassifyEngine.create(ClassifyConfig.builder()
					.modelVersion(name)
					.modelPath(entry.getModelPath())
					.labelsPath(entry.getLabelsPath())
					.resizeShort(entry.getResizeShort())
					.multiLabel(entry.isMultiLabel())
					.topK(entry.getTopK())
					.build());
				engines.put(name, engine);
				log.info("PULC 分类模型已加载: {} -> {}", name, entry.getModelPath());
			}
		}
		return engine;
	}

	/**
	 * 容器关闭时释放全部引擎持有的 ONNX session。
	 */
	@Override
	public void destroy() {
		for (Map.Entry<String, ClassifyEngine> entry : engines.entrySet()) {
			try {
				entry.getValue().close();
			} catch (Exception e) {
				log.warn("关闭 PULC 分类模型 {} 失败: {}", entry.getKey(), e.getMessage());
			}
		}
		engines.clear();
	}
}
