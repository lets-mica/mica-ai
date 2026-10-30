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

import net.dreamlu.mica.ai.enhance.config.EnhanceConfig;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 图像超分引擎池。
 *
 * <p>「处理档位」本质上就是 tile 边长，而 tile 边长在 {@link EnhanceConfig} 构造时就冻结了，
 * 所以每个档位需要独立的 {@link EnhanceEngine}（各自持有一套 ONNX session）。
 * 这里按 tile 边长缓存引擎：首次用到某档位时创建，之后复用，避免每次请求都重新加载模型。
 *
 * <p>池内引擎由本类负责关闭；Spring 容器注入的种子引擎（{@code enhanceEngine}）只用于读取
 * 配置模板，其生命周期仍由容器管理（{@code destroyMethod = "close"}），两者不会重复释放。
 *
 * @author DreamLu
 */
@Component
@ConditionalOnProperty(prefix = "mica.ai.enhance", name = "enabled", havingValue = "true", matchIfMissing = true)
public class EnhanceEnginePool implements DisposableBean {

	private final EnhanceConfig template;

	private final Map<Integer, EnhanceEngine> engines = new ConcurrentHashMap<>();

	/**
	 * 构造引擎池。
	 *
	 * @param seed Spring 容器管理的超分引擎，仅用于读取配置模板
	 */
	public EnhanceEnginePool(EnhanceEngine seed) {
		this.template = seed.getConfig();
	}

	/**
	 * 取指定分块边长对应的引擎，不存在则创建。
	 *
	 * @param tileSize 分块边长（原图像素）
	 * @return 可用的超分引擎
	 */
	public EnhanceEngine engine(int tileSize) {
		return engines.computeIfAbsent(tileSize,
			size -> EnhanceEngine.create(template.toBuilder().tileSize(size).build()));
	}

	/**
	 * 模型原生倍率。
	 *
	 * @return 模型固定的输出倍率
	 */
	public int modelScale() {
		return template.getScale();
	}

	/**
	 * 模型标识。
	 *
	 * @return 模型版本名
	 */
	public String modelVersion() {
		return template.getModelVersion();
	}

	/**
	 * 关闭池内所有引擎。
	 */
	@Override
	public void destroy() {
		engines.values().forEach(EnhanceEngine::close);
		engines.clear();
	}

}
