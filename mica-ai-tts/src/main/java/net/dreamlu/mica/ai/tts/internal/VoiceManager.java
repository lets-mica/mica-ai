package net.dreamlu.mica.ai.tts.internal;

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/**
 * 音色管理器。
 * <p>加载 .bin 格式的音色文件（raw float32，形状 [510, 256]）。
 */
public final class VoiceManager {

	private static final int VOICE_ROWS = 510;
	private static final int VOICE_COLS = 256;
	private static final int VOICE_FLOATS = VOICE_ROWS * VOICE_COLS;

	private final Map<String, float[][]> voiceCache = new HashMap<>();
	private final String voicesDir;

	public VoiceManager(String voicesDir) {
		this.voicesDir = voicesDir;
	}

	/**
	 * 获取指定音色的 style 向量（根据 token 数量索引）。
	 *
	 * @param voiceName  音色名称（如 "zf_001"）
	 * @param tokenCount token 数量（不含 padding）
	 * @return style 向量 [256]
	 */
	public float[] getStyle(String voiceName, int tokenCount) throws IOException {
		float[][] voice = loadVoice(voiceName);
		if (tokenCount < 0 || tokenCount >= VOICE_ROWS) {
			tokenCount = VOICE_ROWS - 1;
		}
		return voice[tokenCount];
	}

	/**
	 * 加载音色文件。
	 */
	private float[][] loadVoice(String voiceName) throws IOException {
		return voiceCache.computeIfAbsent(voiceName, name -> {
			try {
				Path voicePath = Path.of(voicesDir, name + ".bin");
				if (!Files.exists(voicePath)) {
					throw new RuntimeException("Voice file not found: " + voicePath);
				}
				byte[] bytes = Files.readAllBytes(voicePath);
				FloatBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
				float[] flat = new float[VOICE_FLOATS];
				buf.get(flat);
				// reshape to [510, 256]
				float[][] result = new float[VOICE_ROWS][VOICE_COLS];
				for (int i = 0; i < VOICE_ROWS; i++) {
					System.arraycopy(flat, i * VOICE_COLS, result[i], 0, VOICE_COLS);
				}
				return result;
			} catch (IOException e) {
				throw new RuntimeException("Failed to load voice: " + name, e);
			}
		});
	}

	/**
	 * 列出所有可用音色。
	 */
	public List<String> listVoices() throws IOException {
		List<String> voices = new ArrayList<>();
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(Path.of(voicesDir), "*.bin")) {
			for (Path path : stream) {
				String name = path.getFileName().toString();
				voices.add(name.substring(0, name.length() - 4)); // remove .bin
			}
		}
		Collections.sort(voices);
		return voices;
	}
}
