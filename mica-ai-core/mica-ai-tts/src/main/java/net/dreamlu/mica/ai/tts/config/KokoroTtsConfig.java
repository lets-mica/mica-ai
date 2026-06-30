package net.dreamlu.mica.ai.tts.config;

import lombok.Getter;
import net.dreamlu.mica.ai.tts.g2p.G2P;
import net.dreamlu.mica.ai.tts.g2p.HoubbPinyinG2P;

/**
 * Kokoro TTS 配置。
 */
@Getter
public class KokoroTtsConfig {
	/**
	 * 最大音素长度
	 */
	public static final int MAX_PHONEME_LENGTH = 510;
	/**
	 * 采样率
	 */
	public static final int SAMPLE_RATE = 24000;

	private final String modelPath;
	private final String voicesDir;
	private final String configPath;
	private final String defaultVoice;
	private final float defaultSpeed;
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

	public static Builder builder() {
		return new Builder();
	}

	public static class Builder {
		private String modelPath;
		private String voicesDir;
		private String configPath;
		private String defaultVoice = "zf_001";
		private float defaultSpeed = 0.82f;
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
		 * <p>不设置时使用默认的 {@link HoubbPinyinG2P}（基于 houbb/pinyin，多音字消歧、繁简体支持）。
		 * <p>备选：可通过自定义实现 G2P 接口的方式替换，参见 {@link G2P}。
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
			if (g2p == null) {
				this.g2p = new HoubbPinyinG2P();
			}
			return new KokoroTtsConfig(this);
		}
	}
}
