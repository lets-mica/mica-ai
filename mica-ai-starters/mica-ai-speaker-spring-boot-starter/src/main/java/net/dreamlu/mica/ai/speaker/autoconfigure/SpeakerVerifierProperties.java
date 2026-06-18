package net.dreamlu.mica.ai.speaker.autoconfigure;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Speaker Verifier 配置属性。
 *
 * <p>对应 {@code mica.ai.speaker} 配置前缀。
 */
@Data
@ConfigurationProperties(prefix = "mica.ai.speaker")
public class SpeakerVerifierProperties {

	/** 声纹模型路径（必填） */
	private String modelPath;

	/** 默认验证阈值 */
	private float defaultThreshold = 0.58f;

	/** ONNX 内部线程数 */
	private int intraOpNumThreads = 1;

	/** ONNX 交互线程数 */
	private int interOpNumThreads = 1;
}
