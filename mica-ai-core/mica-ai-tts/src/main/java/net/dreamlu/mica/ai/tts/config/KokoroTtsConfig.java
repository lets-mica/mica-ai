package net.dreamlu.mica.ai.tts.config;

import lombok.Getter;
import net.dreamlu.mica.ai.tts.g2p.G2P;
import net.dreamlu.mica.ai.tts.g2p.ChineseG2P;

/**
 * Kokoro TTS 配置。
 */
public class KokoroTtsConfig {
	/**
	 * 最大音素长度
	 */
	public static final int MAX_PHONEME_LENGTH = 510;
	/**
	 * 采样率
	 */
	public static final int SAMPLE_RATE = 24000;

	@Getter
	private final String modelPath;
	@Getter
	private final String voicesDir;
	@Getter
	private final String configPath;
	@Getter
	private final String defaultVoice;
	@Getter
	private final float defaultSpeed;
	@Getter
	private final String onnxProvider;
	private final G2P g2p;

	private KokoroTtsConfig(Builder builder) {
		this.modelPath = builder.modelPath;
		this.voicesDir = builder.voicesDir;
		this.configPath = builder.configPath;
		this.defaultVoice = builder.defaultVoice;
		this.defaultSpeed = builder.defaultSpeed;
		this.onnxProvider = builder.onnxProvider;
		this.g2p = builder.g2p;
	}

	/**
	 * 获取 G2P 转换器。若未设置，返回默认的 {@link ChineseG2P}。
	 */
	public G2P getG2p() {
		return g2p == null ? ChineseG2P.getDefault() : g2p;
	}

	public static Builder builder() {
		return new Builder();
	}

	public static class Builder {
		private String modelPath;
		private String voicesDir;
		private String configPath;
		private String defaultVoice = "zf_001";
		private float defaultSpeed = 1.0f;
		private String onnxProvider = "cpu";
		private G2P g2p;

		public Builder modelPath(String modelPath) {
			this.modelPath = modelPath;
			return this;
		}

		public Builder voicesDir(String voicesDir) {
			this.voicesDir = voicesDir;
			return this;
		}

		public Builder configPath(String configPath) {
			this.configPath = configPath;
			return this;
		}

		public Builder defaultVoice(String defaultVoice) {
			this.defaultVoice = defaultVoice;
			return this;
		}

		public Builder defaultSpeed(float defaultSpeed) {
			this.defaultSpeed = defaultSpeed;
			return this;
		}

		public Builder onnxProvider(String onnxProvider) {
			this.onnxProvider = onnxProvider;
			return this;
		}

		/**
		 * 注入自定义 G2P 转换器。
		 * <p>不设置时使用 {@link ChineseG2P} 简化实现（仅覆盖 ~80 个常用汉字）。
		 * <p>推荐使用 {@code net.dreamlu.mica.ai.tts.g2p.HoubbPinyinG2P}（需自行引入 houbb/pinyin 依赖）。
		 *
		 * @param g2p G2P 实例
		 */
		public Builder g2p(G2P g2p) {
			this.g2p = g2p;
			return this;
		}

		public KokoroTtsConfig build() {
			if (modelPath == null) {
				throw new IllegalArgumentException("modelPath is required");
			}
			if (voicesDir == null) {
				throw new IllegalArgumentException("voicesDir is required");
			}
			if (configPath == null) {
				throw new IllegalArgumentException("configPath is required");
			}
			return new KokoroTtsConfig(this);
		}
	}
}
