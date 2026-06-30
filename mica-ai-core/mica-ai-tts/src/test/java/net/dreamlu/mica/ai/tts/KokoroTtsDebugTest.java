package net.dreamlu.mica.ai.tts;

import net.dreamlu.mica.ai.tts.config.KokoroTtsConfig;
import net.dreamlu.mica.ai.tts.config.TtsResult;
import net.dreamlu.mica.ai.tts.engine.KokoroTts;
import org.junit.jupiter.api.Test;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 诊断性测试：保留 WAV 文件以便与 Python 参考实现对比。
 * 仅在模型文件存在时运行。
 */
class KokoroTtsDebugTest {

	private static final String MODEL_DIR = "E:\\codes\\ai\\kokoro-onnx\\model";
	private static final String OUT_DIR = "E:\\codes\\ai\\mica-ai\\model-tools";

	@Test
	void dumpWavForComparison() throws Exception {
		Path modelPath = Path.of(MODEL_DIR, "model_dynamic.onnx");
		Path voicesDir = Path.of(MODEL_DIR, "voices");
		Path configPath = Path.of(MODEL_DIR, "config.json");
		if (!Files.exists(modelPath) || !Files.exists(configPath)) {
			System.out.println("Model files not found, skipping");
			return;
		}

		KokoroTtsConfig config = KokoroTtsConfig.builder()
			.modelPath(modelPath.toString())
			.voicesDir(voicesDir.toString())
			.configPath(configPath.toString())
			.defaultVoice("zf_001")
			.defaultSpeed(1.0f)
			.build();

		// 与 Python probe_java_v4.wav 同样的输入：4 个真实音素
		String phonemes = "ㄋㄧ ㄏㄠ ㄨㄛ ㄕ";
		try (KokoroTts tts = new KokoroTts(config)) {
			TtsResult result = tts.synthesizeFromPhonemes(phonemes, "zf_001", 1.0f);
			System.out.printf("Java  audio_len=%d  min=%.4f  max=%.4f  mean_abs=%.4f%n",
				result.audio().length,
				min(result.audio()),
				max(result.audio()),
				meanAbs(result.audio()));

			Path out = Path.of(OUT_DIR, "probe_java_mica.wav");
			saveWav(result.audio(), 24000, out);
			System.out.println("Wrote " + out + " (" + Files.size(out) + " bytes)");

			// 第二个测试：使用 KokoroTtsTest.testSynthesizeWithPhonemes 中的输入
			String phonemes2 = "ㄋㄧ3 ㄏㄠ3 ㄨㄛ3 ㄕ4 Kokoro";
			TtsResult result2 = tts.synthesizeFromPhonemes(phonemes2, "zf_001", 1.0f);
			System.out.printf("Java2 audio_len=%d  min=%.4f  max=%.4f  mean_abs=%.4f%n",
				result2.audio().length,
				min(result2.audio()),
				max(result2.audio()),
				meanAbs(result2.audio()));
			Path out2 = Path.of(OUT_DIR, "probe_java_mica_v2.wav");
			saveWav(result2.audio(), 24000, out2);
			System.out.println("Wrote " + out2 + " (" + Files.size(out2) + " bytes)");
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
