/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtProvider;
import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;

/**
 * ONNX Runtime execution provider 自动选择。
 *
 * <ul>
 *   <li>{@code preferCpu=true} → 强制使用 {@code CPUExecutionProvider}（保证跨平台 bit-exact 精度）</li>
 *   <li>{@code preferCpu=false} → 按 CoreML (macOS) > CUDA > CPU 自动选择</li>
 * </ul>
 */
@Slf4j
@UtilityClass
public class OrtProviders {

	/**
	 * @param preferCpu {@code true} 强制 CPU；{@code false} 自动选择加速器
	 * @return ONNX Runtime provider 名称列表
	 */
	public static String[] resolve(boolean preferCpu) {
		if (preferCpu) {
			log.info("ONNX Runtime provider: CPUExecutionProvider (forced)");
			return new String[]{"CPUExecutionProvider"};
		}
		EnumSet<OrtProvider> available;
		try {
			OrtEnvironment.getEnvironment();
			available = OrtEnvironment.getAvailableProviders();
		} catch (Exception e) {
			log.warn("无法枚举 ONNX Runtime providers, 回退到 CPU: {}", e.getMessage());
			return new String[]{"CPUExecutionProvider"};
		}
		List<String> availableNames = new ArrayList<>(available.size());
		for (OrtProvider p : available) {
			availableNames.add(p.getName());
		}
		for (String preferred : List.of("CoreMLExecutionProvider", "CUDAExecutionProvider")) {
			if (availableNames.contains(preferred)) {
				log.info("ONNX Runtime provider: {}", preferred);
				return new String[]{preferred};
			}
		}
		log.info("ONNX Runtime provider: CPUExecutionProvider (fallback)");
		return new String[]{"CPUExecutionProvider"};
	}
}
