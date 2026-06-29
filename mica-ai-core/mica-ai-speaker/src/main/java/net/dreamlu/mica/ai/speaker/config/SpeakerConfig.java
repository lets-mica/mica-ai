package net.dreamlu.mica.ai.speaker.config;

import ai.onnxruntime.OrtSession;
import lombok.Builder;
import lombok.Getter;

/**
 * Speaker Verifier 引擎配置。
 *
 * <p>使用 Builder 模式构建，所有参数均有合理默认值：
 * <pre>{@code
 * SpeakerConfig config = SpeakerConfig.builder()
 *     .modelPath("models/eres2net.onnx")
 *     .threshold(0.58f)
 *     .build();
 *
 * try (SpeakerVerifier verifier = new SpeakerVerifier(config)) {
 *     // ...
 * }
 * }</pre>
 */
@Getter
@Builder
public final class SpeakerConfig {

	/** 声纹模型 ONNX 路径（必填） */
	private String modelPath;

	/** 默认验证阈值，影响 {@link net.dreamlu.mica.ai.speaker.engine.SpeakerVerifier#verify(float[], java.nio.file.Path)} 的判分，默认 0.58 */
	@Builder.Default
	private float threshold = 0.58f;

	/** ONNX Runtime 内部线程数，默认 1 */
	@Builder.Default
	private int intraOpNumThreads = 1;

	/** ONNX Runtime 交互线程数，默认 1 */
	@Builder.Default
	private int interOpNumThreads = 1;

	/** ONNX Runtime 图优化级别，默认 {@link OrtSession.SessionOptions.OptLevel#BASIC_OPT} */
	@Builder.Default
	private OrtSession.SessionOptions.OptLevel optimizationLevel = OrtSession.SessionOptions.OptLevel.BASIC_OPT;

}
