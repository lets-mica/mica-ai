package net.dreamlu.mica.ai.voice.autoconfigure;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.ArrayList;
import java.util.List;

/**
 * SenseVoice 配置属性。
 *
 * <p>对应 {@code mica.ai.voice} 配置前缀。
 */
@Data
@ConfigurationProperties(prefix = "mica.ai.voice")
public class SenseVoiceProperties {

	/**
	 * 是否启用该 Starter。默认 {@code true}：启用时必填的 encoder/decoder/tokenizer 路径缺失将启动失败；
	 * 设为 {@code false} 时整个 Starter 不注入任何 Bean。
	 */
	private boolean enabled = true;

	/** 编码器模型路径（必填） */
	private String encoderPath;

	/** 解码器模型路径（必填） */
	private String decoderPath;

	/** 分词器模型路径（必填） */
	private String tokenizerPath;

	/** ONNX Runtime Provider: cpu / cuda / dml */
	private String onnxProvider = "cpu";

	/** 是否启用 ITN 数字规范化 */
	private boolean itn = true;

	/** CTC 解码 Top-K */
	private int topK = 10;

	/** 热词列表 */
	private List<String> hotwords = new ArrayList<>();
}
