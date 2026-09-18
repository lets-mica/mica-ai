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
package net.dreamlu.mica.ai.common.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * ONNX Runtime 执行提供器（Execution Provider）自动选择与注册。
 *
 * <ul>
 *   <li>{@code device = CPU} → 强制 CPU，跨平台 bit-exact</li>
 *   <li>{@code device = GPU} → 按 CoreML (macOS) &gt; CUDA &gt; CPU 自动选择最佳加速器</li>
 * </ul>
 *
 * <p>注册失败仅 warn，回退到 CPU，{@link OrtSession.SessionOptions} 仍可继续创建。
 */
@Slf4j
@UtilityClass
public class OrtProviders {

	private static final String CPU = "CPUExecutionProvider";
	private static final String CUDA = "CUDAExecutionProvider";
	private static final String CORE_ML = "CoreMLExecutionProvider";

	@FunctionalInterface
	private interface EpRegistrar {
		void register(OrtSession.SessionOptions opts, int deviceId) throws OrtException;
	}

	private static final Map<String, EpRegistrar> REGISTRARS;

	static {
		Map<String, EpRegistrar> map = new LinkedHashMap<>();
		map.put(CUDA, OrtSession.SessionOptions::addCUDA);
		map.put(CORE_ML, (opts, deviceId) -> opts.addCoreML());
		REGISTRARS = Collections.unmodifiableMap(map);
	}

	/**
	 * 解析当前运行时可用的 ONNX Runtime provider 名称（不注册）。
	 *
	 * @param preferCpu true 强制 CPU；false 按 CoreML &gt; CUDA 自动选
	 * @return provider 名称数组，首个元素为最终选择；返回数组永不为 null，
	 *         长度为 1，枚举失败时回退 {@code CPU}
	 */
	public static String[] resolve(boolean preferCpu) {
		if (preferCpu) {
			log.info("mica-ai: ONNX Runtime provider forced to {}", CPU);
			return new String[]{CPU};
		}
		List<String> available;
		try {
			EnumSet<OrtProvider> set = OrtEnvironment.getAvailableProviders();
			available = new ArrayList<>(set == null ? 0 : set.size());
			if (set != null) {
				for (OrtProvider p : set) {
					available.add(p.getName());
				}
			}
		} catch (Exception e) {
			log.warn("mica-ai: 无法枚举 ONNX Runtime providers，回退 CPU: {}", e.getMessage());
			return new String[]{CPU};
		}
		for (String preferred : new String[]{CORE_ML, CUDA}) {
			if (available.contains(preferred) && REGISTRARS.containsKey(preferred)) {
				log.info("mica-ai: ONNX Runtime provider auto-selected: {}", preferred);
				return new String[]{preferred};
			}
		}
		log.info("mica-ai: ONNX Runtime provider fallback to {}", CPU);
		return new String[]{CPU};
	}

	/**
	 * 把 {@code providers[0]} 解析到的加速器注册到 {@link OrtSession.SessionOptions}。
	 *
	 * <p>注册失败仅 warn（不抛异常），由调用方继续走默认 CPU 推理；空数组 / null
	 * 直接 no-op。
	 *
	 * @param providers   {@link #resolve(boolean)} 返回的 provider 名数组
	 * @param opts        待配置的 SessionOptions
	 * @param deviceId    GPU device id（CUDA 专用，CoreML 忽略）
	 */
	public static void apply(String[] providers, OrtSession.SessionOptions opts, int deviceId) {
		if (providers == null || providers.length == 0) {
			return;
		}
		String name = providers[0];
		EpRegistrar registrar = REGISTRARS.get(name);
		if (registrar == null) {
			return;
		}
		try {
			registrar.register(opts, deviceId);
			log.info("mica-ai: 已注册 ONNX Runtime provider: {} (deviceId={})", name, deviceId);
		} catch (OrtException e) {
			log.warn("mica-ai: 注册 {} 失败，回退 CPU: {}", name, e.getMessage());
		}
	}
}
