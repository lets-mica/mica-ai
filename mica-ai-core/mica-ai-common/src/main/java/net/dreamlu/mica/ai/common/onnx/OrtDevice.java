package net.dreamlu.mica.ai.common.onnx;

/**
 * ONNX Runtime 执行设备选项。
 *
 * <ul>
 *   <li>{@link #CPU} —— 跨平台 bit-exact，默认</li>
 *   <li>{@link #GPU} —— 由 {@link OrtProviders} 按 CoreML &gt; CUDA 顺序自动选择；
 *       运行时无对应 EP 时自动回退 CPU 并打 warn</li>
 * </ul>
 */
public enum OrtDevice {
	CPU,
	GPU
}
