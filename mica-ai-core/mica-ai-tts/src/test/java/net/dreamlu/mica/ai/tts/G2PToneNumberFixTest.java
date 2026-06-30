package net.dreamlu.mica.ai.tts;

import net.dreamlu.mica.ai.tts.config.KokoroTtsConfig;
import net.dreamlu.mica.ai.tts.config.TtsResult;
import net.dreamlu.mica.ai.tts.engine.KokoroTts;
import net.dreamlu.mica.ai.tts.g2p.HoubbPinyinG2P;
import org.junit.jupiter.api.Test;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.FileWriter;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 验证修复后的 G2P 输出包含声调数字 token，并端到端 TTS 验证。
 */
class G2PToneNumberFixTest {

	private static final String MODEL_DIR = "E:\\codes\\ai\\kokoro-onnx\\model";
	private static final String OUT_DIR = "E:\\codes\\ai\\mica-ai";

	@Test
	void verifyG2PIncludesToneNumbers() {
		HoubbPinyinG2P g2p = new HoubbPinyinG2P();

		// "你好" 期望输出: "ㄋㄧ3 ㄏㄠ3"（带声调数字）
		String niHao = g2p.convert("你好");
		System.out.println("niHao = " + niHao);
		assertTrue(niHao.contains("3"),
			"phonemes should contain tone number 3 (for 你好), got: " + niHao);

		// "欢迎" 期望: "ㄏㄨㄢ1 ㄧㄥ2"
		String huanYing = g2p.convert("欢迎");
		System.out.println("huanYing = " + huanYing);
		assertTrue(huanYing.contains("1") || huanYing.contains("2"),
			"phonemes should contain tone numbers, got: " + huanYing);

		// 完整句子
		String full = g2p.convert("你好，欢迎使用 mica-ai。");
		System.out.println("full = " + full);

		// 端到端 TTS 测试
		Path modelPath = Path.of(MODEL_DIR, "model_dynamic.onnx");
		Path voicesDir = Path.of(MODEL_DIR, "voices");
		Path configPath = Path.of(MODEL_DIR, "config.json");
		if (!Files.exists(modelPath) || !Files.exists(configPath)) {
			System.out.println("Model files not found, skipping E2E TTS");
			return;
		}

		KokoroTtsConfig config = KokoroTtsConfig.builder()
			.modelPath(modelPath.toString())
			.voicesDir(voicesDir.toString())
			.configPath(configPath.toString())
			.defaultVoice("zf_001")
			.defaultSpeed(1.0f)
			.g2p(g2p)
			.build();

		try (KokoroTts tts = new KokoroTts(config)) {
			TtsResult result = tts.synthesize("你好，欢迎使用 mica-ai。", "zf_001", 1.0f);
			System.out.printf("audio_len=%d  duration=%.3fs  min=%.4f  max=%.4f  mean_abs=%.4f%n",
				result.audio().length, result.duration(),
				min(result.audio()), max(result.audio()), meanAbs(result.audio()));

			assertTrue(result.audio().length > 0, "audio should be non-empty");
			assertTrue(max(result.audio()) > 0.0f, "audio max should be positive (no NaN)");
			assertTrue(min(result.audio()) < 0.0f, "audio min should be negative (no NaN)");

			Path out = Path.of(OUT_DIR, "user_input_fixed.wav");
			saveWav(result.audio(), 24000, out);
			System.out.println("Wrote " + out + " (" + Files.size(out) + " bytes)");

			// 同步写入 phonemes 信息
			try (PrintWriter pw = new PrintWriter(new FileWriter("E:\\codes\\ai\\mica-ai\\user_input_fixed.txt"))) {
				pw.println("=== INPUT ===");
				pw.println("你好，欢迎使用 mica-ai。");
				pw.println("=== PHONEMES ===");
				pw.println(full);
				pw.println();
				pw.printf("audio_len=%d  duration=%.3fs%n", result.audio().length, result.duration());
			}
		} catch (Exception e) {
			fail("E2E TTS failed: " + e.getMessage(), e);
		}
	}

	private static float min(float[] a) {
		float m = Float.POSITIVE_INFINITY;
		for (float v : a) if (v < m) m = v;
		return m;
	}

	private static float max(float[] a) {
		float m = Float.NEGATIVE_INFINITY;
		for (float v : a) if (v > m) m = v;
		return m;
	}

	private static float meanAbs(float[] a) {
		double s = 0;
		for (float v : a) s += Math.abs(v);
		return (float) (s / a.length);
	}

	private static void saveWav(float[] audio, int sampleRate, Path file) throws IOException {
		try (OutputStream os = Files.newOutputStream(file);
			 DataOutputStream dos = new DataOutputStream(os)) {
			dos.writeBytes("RIFF");
			dos.writeInt(Integer.reverseBytes(36 + audio.length * 2));
			dos.writeBytes("WAVE");
			dos.writeBytes("fmt ");
			dos.writeInt(Integer.reverseBytes(16));
			dos.writeShort(Short.reverseBytes((short) 1));
			dos.writeShort(Short.reverseBytes((short) 1));
			dos.writeInt(Integer.reverseBytes(sampleRate));
			dos.writeInt(Integer.reverseBytes(sampleRate * 2));
			dos.writeShort(Short.reverseBytes((short) 2));
			dos.writeShort(Short.reverseBytes((short) 16));
			dos.writeBytes("data");
			dos.writeInt(Integer.reverseBytes(audio.length * 2));
			for (float sample : audio) {
				float clamped = Math.max(-1f, Math.min(1f, sample));
				short pcm = (short) (clamped * 32767f);
				dos.writeShort(Short.reverseBytes(pcm));
			}
		}
	}
}
