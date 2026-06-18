package net.dreamlu.mica.ai.voice.config;

import java.util.List;

/**
 * SenseVoice 引擎配置参数。
 *
 * <p>使用 Builder 风格链式调用构造实例：
 * <pre>{@code
 * SenseVoiceConfig config = new SenseVoiceConfig()
 *     .encoderPath("encoder.onnx")
 *     .decoderPath("decoder.onnx")
 *     .tokenizerPath("tokenizer.model")
 *     .hotWords(List.of("mica", "梦想卢"))
 *     .topK(10);
 * }</pre>
 */
public final class SenseVoiceConfig {

	private String encoderPath;
	private String decoderPath;
	private String tokenizerPath;
	private String onnxProvider = "cpu";
	private List<String> hotwords;
	private int topK = 10;
	private boolean itn = true;

	public String getEncoderPath() {
		return encoderPath;
	}

	public SenseVoiceConfig encoderPath(String encoderPath) {
		this.encoderPath = encoderPath;
		return this;
	}

	public String getDecoderPath() {
		return decoderPath;
	}

	public SenseVoiceConfig decoderPath(String decoderPath) {
		this.decoderPath = decoderPath;
		return this;
	}

	public String getTokenizerPath() {
		return tokenizerPath;
	}

	public SenseVoiceConfig tokenizerPath(String tokenizerPath) {
		this.tokenizerPath = tokenizerPath;
		return this;
	}

	public String getOnnxProvider() {
		return onnxProvider;
	}

	/**
	 * 设置 ONNX Runtime 推理后端。
	 *
	 * @param onnxProvider 可选值: "cpu", "cuda", "dml"
	 */
	public SenseVoiceConfig onnxProvider(String onnxProvider) {
		this.onnxProvider = onnxProvider;
		return this;
	}

	public List<String> getHotwords() {
		return hotwords;
	}

	public SenseVoiceConfig hotwords(List<String> hotwords) {
		this.hotwords = hotwords;
		return this;
	}

	public int getTopK() {
		return topK;
	}

	/**
	 * 设置热词搜索 Top-K 深度。
	 *
	 * @param topK 默认 10
	 */
	public SenseVoiceConfig topK(int topK) {
		this.topK = topK;
		return this;
	}

	public boolean isItn() {
		return itn;
	}

	/**
	 * 是否启用反向文本规范化（ITN）。
	 *
	 * @param itn 默认 true
	 */
	public SenseVoiceConfig itn(boolean itn) {
		this.itn = itn;
		return this;
	}
}
