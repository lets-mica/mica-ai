/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.MicaAiException;

import java.util.Set;

/**
 * 将 {@link OrtSessionOptions} 翻译为 ONNX Runtime 原生 {@link OrtSession.SessionOptions}。
 */
@Slf4j
@UtilityClass
public class OrtSessionFactory {

	public static OrtSession.SessionOptions build(OrtSessionOptions cfg) {
		OrtSessionOptions options = cfg != null ? cfg : OrtSessionOptions.defaults();
		OrtSession.SessionOptions so = new OrtSession.SessionOptions();
		try {
			if (options.getIntraOpNumThreads() > 0) {
				so.setIntraOpNumThreads(options.getIntraOpNumThreads());
			}
			if (options.getInterOpNumThreads() > 0) {
				so.setInterOpNumThreads(options.getInterOpNumThreads());
			}
			so.setOptimizationLevel(mapLevel(options.getGraphOptimizationLevel()));
			so.setExecutionMode(mapMode(options.getExecutionMode()));
		} catch (OrtException e) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.MODEL_LOAD_FAILED, "配置 ONNX 会话选项失败", e);
		}

		if (options.getDevice() == OrtSessionOptions.Device.GPU) {
			applyCuda(so, options.getCudaDeviceId());
		}
		return so;
	}

	private static void applyCuda(OrtSession.SessionOptions so, int deviceId) {
		try {
			Set<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
			if (providers == null || !providers.contains(OrtProvider.CUDA)) {
				log.warn("mica-ai-face: 未检测到可用的 CUDA 执行提供器，回退到 CPU 推理");
				return;
			}
			so.addCUDA(deviceId);
			log.info("mica-ai-face: 已启用 CUDA 执行提供器 (deviceId={})", deviceId);
		} catch (OrtException e) {
			log.warn("mica-ai-face: 启用 CUDA 执行提供器失败，回退到 CPU 推理: {}", e.getMessage());
		}
	}

	private static OptLevel mapLevel(OrtSessionOptions.GraphOptimizationLevel level) {
		if (level == null) {
			return OptLevel.ALL_OPT;
		}
		switch (level) {
			case ORT_DISABLE_ALL:
				return OptLevel.NO_OPT;
			case ORT_ENABLE_BASIC:
				return OptLevel.BASIC_OPT;
			case ORT_ENABLE_EXTENDED:
				return OptLevel.EXTENDED_OPT;
			case ORT_ENABLE_ALL:
			default:
				return OptLevel.ALL_OPT;
		}
	}

	private static ExecutionMode mapMode(OrtSessionOptions.ExecutionMode mode) {
		if (mode == null) {
			return ExecutionMode.PARALLEL;
		}
		switch (mode) {
			case ORT_SEQUENTIAL:
				return ExecutionMode.SEQUENTIAL;
			case ORT_PARALLEL:
			default:
				return ExecutionMode.PARALLEL;
		}
	}
}