/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;

import java.util.Set;

/**
 * 将 {@link OrtSessionOptions} 配置转换为 ONNX Runtime 原生的
 * {@link OrtSession.SessionOptions}，并按设备选择执行提供器（Execution Provider）。
 *
 * @author L.cm
 */
@Slf4j
@UtilityClass
public class OrtSessionFactory {

	/**
	 * 根据配置构建 {@link OrtSession.SessionOptions}。
	 *
	 * <p>线程数 / 图优化级别 / 执行模式均按配置设置；当 {@code device = GPU} 时，
	 * 先检测运行环境是否真的提供 CUDA 执行提供器，仅有才挂载，否则回退 CPU 并打印告警，
	 * 避免在没有 GPU 的机器上启动即崩溃。
	 *
	 * @param cfg 基础配置，{@code null} 时按默认处理
	 * @return 构建好的会话选项，调用方负责在所属的 {@code ModelManager} 销毁时 close 它
	 */
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
				ErrorCode.MODEL_LOAD_FAILED, "配置 ONNX 会话选项失败", e);
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
				log.warn("mica-ai: 未检测到可用的 CUDA 执行提供器，回退到 CPU 推理");
				return;
			}
			so.addCUDA(deviceId);
			log.info("mica-ai: 已启用 CUDA 执行提供器 (deviceId={})", deviceId);
		} catch (OrtException e) {
			log.warn("mica-ai: 启用 CUDA 执行提供器失败，回退到 CPU 推理: {}", e.getMessage());
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