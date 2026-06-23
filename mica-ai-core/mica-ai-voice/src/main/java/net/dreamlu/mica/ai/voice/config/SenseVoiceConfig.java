package net.dreamlu.mica.ai.voice.config;

import lombok.Builder;
import lombok.Getter;

import java.util.List;

/**
 * SenseVoice 引擎配置参数。
 *
 * <p>使用 Builder 模式构建，所有参数均有合理默认值。
 *
 * <p>示例：
 * <pre>{@code
 * SenseVoiceConfig config = SenseVoiceConfig.builder()
 *     .encoderPath("encoder.onnx")
 *     .decoderPath("decoder.onnx")
 *     .tokenizerPath("tokenizer.model")
 *     .hotwords(List.of("mica", "梦想卢"))
 *     .topK(10)
 *     .build();
 * }</pre>
 */
@Getter
@Builder
public final class SenseVoiceConfig {

	/** 编码器 ONNX 模型路径（必填） */
	private String encoderPath;

	/** 解码器 ONNX 模型路径（必填） */
	private String decoderPath;

	/** BPE tokenizer 模型路径（必填） */
	private String tokenizerPath;

	/** ONNX Runtime 推理后端，默认 {@code cpu}；可选 {@code cpu} / {@code cuda} / {@code dml} */
	@Builder.Default
	private String onnxProvider = "cpu";

	/** 热词列表，按 topK 深度匹配 CTC 输出前缀；可空 */
	private List<String> hotwords;

	/** 热词搜索 Top-K 深度，默认 10 */
	@Builder.Default
	private int topK = 10;

	/** 是否启用反向文本规范化（ITN），默认 true */
	@Builder.Default
	private boolean itn = true;

	public static SenseVoiceConfig defaults() {
		return builder().build();
	}
}
