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

	/** ONNX 模型文件路径（必填） */
	private String modelPath;

	/** 音色文件目录路径（必填） */
	private String voicesDir;

	/** 配置文件路径（必填） */
	private String configPath;

	/** 默认音色 */
	private String defaultVoice = "zf_001";

	/** 默认语速 */
	private float defaultSpeed = 1.0f;

	/** ONNX Runtime Provider: cpu / cuda */
	private String onnxProvider = "cpu";
}
