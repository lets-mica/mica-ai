package net.dreamlu.mica.ai.tts;

import net.dreamlu.mica.ai.tts.g2p.ChineseG2P;
import net.dreamlu.mica.ai.tts.g2p.G2P;
import net.dreamlu.mica.ai.tts.internal.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Kokoro TTS 纯 ONNXRuntime 语音合成引擎。
 * <p>基于 Kokoro-82M 模型，支持中英双语、多音色离线语音合成。
 *
 * <p>用法：
 * <pre>{@code
 * KokoroTtsConfig config = KokoroTtsConfig.builder()
 *     .modelPath("model/model_dynamic.onnx")
 *     .voicesDir("model/voices")
 *     .configPath("model/config.json")
 *     .defaultVoice("zf_001")
 *     .build();
 *
 * try (KokoroTts tts = new KokoroTts(config)) {
 *     TtsResult result = tts.synthesize("你好世界");
 *     tts.saveWav(result, "output.wav");
 * }
 * }</pre>
 */
public class KokoroTts implements AutoCloseable {

	private static final Logger log = LoggerFactory.getLogger(KokoroTts.class);
	private static final int SAMPLE_RATE = 24000;

	private final KokoroTtsConfig config;
	private final KokoroEngine engine;
	private final Vocab vocab;
	private final VoiceManager voiceManager;
	private final G2P g2p;

	public KokoroTts(KokoroTtsConfig config) throws Exception {
		this.config = config;
		this.engine = new KokoroEngine(config.getModelPath(), config.getOnnxProvider());
		this.vocab = Vocab.load(config.getConfigPath());
		this.voiceManager = new VoiceManager(config.getVoicesDir());
		this.g2p = config.getG2p();
		log.info("KokoroTts initialized with model: {} (G2P: {})",
			config.getModelPath(), g2p.getClass().getSimpleName());
	}

	/**
	 * 合成语音。
	 *
	 * @param text 输入文本（支持中文、英文、数字混合）
	 * @return 合成结果
	 */
	public TtsResult synthesize(String text) throws Exception {
		return synthesize(text, config.getDefaultVoice(), config.getDefaultSpeed());
	}

	/**
	 * 合成语音（指定音色和语速）。
	 *
	 * @param text  输入文本
	 * @param voice 音色名称
	 * @param speed 语速（0.5 ~ 2.0）
	 * @return 合成结果
	 */
	public TtsResult synthesize(String text, String voice, float speed) throws Exception {
		if (text == null || text.trim().isEmpty()) {
			return new TtsResult(new float[0], SAMPLE_RATE, 0);
		}

		speed = Math.max(0.5f, Math.min(2.0f, speed));

		String phonemes = g2p(text);
		log.debug("Phonemes: {}", phonemes);

		if (phonemes.isEmpty()) {
			return new TtsResult(new float[0], SAMPLE_RATE, 0);
		}

		List<String> batches = TextFrontend.splitPhonemes(phonemes);
		log.debug("Split into {} batches", batches.size());

		List<float[]> audioChunks = new ArrayList<>();
		for (String batch : batches) {
			float[] chunk = synthesizeBatch(batch, voice, speed);
			chunk = TextFrontend.trimSilence(chunk, 0.01f);
			audioChunks.add(chunk);
		}

		float[] audio = concatAudio(audioChunks);
		double duration = (double) audio.length / SAMPLE_RATE;

		return new TtsResult(audio, SAMPLE_RATE, duration);
	}

	/**
	 * 从预生成的音素合成语音。
	 */
	public TtsResult synthesizeFromPhonemes(String phonemes, String voice, float speed) throws Exception {
		if (phonemes == null || phonemes.isEmpty()) {
			return new TtsResult(new float[0], SAMPLE_RATE, 0);
		}

		speed = Math.max(0.5f, Math.min(2.0f, speed));
		float[] audio = synthesizeBatch(phonemes, voice, speed);
		audio = TextFrontend.trimSilence(audio, 0.01f);
		double duration = (double) audio.length / SAMPLE_RATE;
		return new TtsResult(audio, SAMPLE_RATE, duration);
	}

	/**
	 * 列出所有可用音色。
	 */
	public List<String> listVoices() throws IOException {
		return voiceManager.listVoices();
	}

	/**
	 * 保存音频为 WAV 文件（PCM 16-bit）。
	 */
	public void saveWav(TtsResult result, String filePath) throws IOException {
		writeWav(result.audio(), result.sampleRate(), filePath);
	}

	/**
	 * G2P：文本转音素（使用注入的 G2P）。
	 * <p>默认使用 {@link ChineseG2P} 简化实现。
	 * 可通过 {@link KokoroTtsConfig.Builder#g2p(G2P)} 注入自定义实现（如 houbb/pinyin）。
	 */
	private String g2p(String text) {
		return g2p.convert(text);
	}

	/**
	 * 单批次合成。
	 */
	private float[] synthesizeBatch(String phonemes, String voice, float speed) throws Exception {
		// 过滤掉词表中不存在的字符
		String filtered = vocab.filter(phonemes);
		if (filtered.isEmpty()) {
			return new float[0];
		}

		// 截断超过最大长度的音素
		if (filtered.length() > KokoroTtsConfig.MAX_PHONEME_LENGTH) {
			filtered = filtered.substring(0, KokoroTtsConfig.MAX_PHONEME_LENGTH);
		}

		// 转换为 token ID
		List<Integer> tokenList = vocab.tokenize(filtered);
		if (tokenList.isEmpty()) {
			return new float[0];
		}

		// 添加首尾 padding 0
		long[] inputIds = new long[tokenList.size() + 2];
		inputIds[0] = 0;
		for (int i = 0; i < tokenList.size(); i++) {
			inputIds[i + 1] = tokenList.get(i);
		}
		inputIds[inputIds.length - 1] = 0;

		// 获取 style 向量
		float[] refS = voiceManager.getStyle(voice, tokenList.size());

		// 推理
		return engine.inference(inputIds, refS, speed);
	}

	/**
	 * 拼接音频块。
	 */
	private float[] concatAudio(List<float[]> chunks) {
		int total = 0;
		for (float[] c : chunks) {
			total += c.length;
		}
		float[] result = new float[total];
		int offset = 0;
		for (float[] c : chunks) {
			System.arraycopy(c, 0, result, offset, c.length);
			offset += c.length;
		}
		return result;
	}

	/**
	 * 写入 WAV 文件（PCM 16-bit）。
	 */
	private void writeWav(float[] audio, int sampleRate, String filePath) throws IOException {
		try (DataOutputStream dos = new DataOutputStream(
			new BufferedOutputStream(Files.newOutputStream(Paths.get(filePath))))) {
			// RIFF header
			dos.writeBytes("RIFF");
			dos.writeInt(Integer.reverseBytes(36 + audio.length * 2)); // file size - 8
			dos.writeBytes("WAVE");

			// fmt chunk
			dos.writeBytes("fmt ");
			dos.writeInt(Integer.reverseBytes(16)); // chunk size
			dos.writeShort(Short.reverseBytes((short) 1)); // PCM format
			dos.writeShort(Short.reverseBytes((short) 1)); // mono
			dos.writeInt(Integer.reverseBytes(sampleRate));
			dos.writeInt(Integer.reverseBytes(sampleRate * 2)); // byte rate
			dos.writeShort(Short.reverseBytes((short) 2)); // block align
			dos.writeShort(Short.reverseBytes((short) 16)); // bits per sample

			// data chunk
			dos.writeBytes("data");
			dos.writeInt(Integer.reverseBytes(audio.length * 2));

			for (float sample : audio) {
				// Clamp 到 [-1, 1] 范围
				float clamped = Math.max(-1f, Math.min(1f, sample));
				short pcm = (short) (clamped * 32767f);
				dos.writeShort(Short.reverseBytes(pcm));
			}
		}
	}

	@Override
	public void close() throws Exception {
		if (engine != null) {
			engine.close();
		}
	}
}
