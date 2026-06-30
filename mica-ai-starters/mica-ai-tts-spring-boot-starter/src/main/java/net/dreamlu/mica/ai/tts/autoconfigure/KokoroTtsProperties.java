package net.dreamlu.mica.ai.tts.autoconfigure;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Kokoro TTS 配置属性。
 *
 * <p>对应 {@code mica.ai.tts} 配置前缀。
 */
@Data
@ConfigurationProperties(prefix = "mica.ai.tts")
public class KokoroTtsProperties {

	/**
	 * 是否启用该 Starter。
	 *
	 * <p>默认 {@code true}：启用时如果必填的模型/词表/音色路径未配置，会在启动时 fail-fast 抛出异常；
	 * 设为 {@code false} 时整个 Starter 不注入任何 Bean，可安全留空。
	 */
	private boolean enabled = true;

	/** ONNX 模型文件路径（必填） */
	private String modelPath;

	/** 音色文件目录路径（必填） */
	private String voicesDir;

	/** 配置文件路径（必填） */
	private String configPath;

	/** 默认音色 */
	private String defaultVoice = "zf_001";

	/** 默认语速 */
	private float defaultSpeed = 0.82f;

	/** ONNX Runtime Provider: cpu / cuda */
	private String onnxProvider = "cpu";
}
