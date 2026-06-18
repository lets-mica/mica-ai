package net.dreamlu.mica.ai.tts;

import net.dreamlu.mica.ai.tts.config.KokoroTtsConfig;
import net.dreamlu.mica.ai.tts.config.TtsResult;
import net.dreamlu.mica.ai.tts.engine.KokoroTts;
import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * KokoroTts 端到端测试。
 * <p>需要已下载 Kokoro-82M ONNX 模型和音色文件到 {@code E:\codes\ai\kokoro-onnx\model}。
 */
class KokoroTtsTest {

	private static final String MODEL_DIR = "E:\\codes\\ai\\kokoro-onnx\\model";

	@Test
	void testSynthesizeWithPhonemes() throws Exception {
		Path modelPath = Path.of(MODEL_DIR, "model_dynamic.onnx");
		Path voicesDir = Path.of(MODEL_DIR, "voices");
		Path configPath = Path.of(MODEL_DIR, "config.json");

		if (!Files.exists(modelPath) || !Files.exists(configPath)) {
			System.out.println("Model files not found, skipping test");
			return;
		}

		KokoroTtsConfig config = KokoroTtsConfig.builder()
			.modelPath(modelPath.toString())
			.voicesDir(voicesDir.toString())
			.configPath(configPath.toString())
			.defaultVoice("zf_001")
			.defaultSpeed(1.0f)
			.build();

		try (KokoroTts tts = new KokoroTts(config)) {
			// 使用预生成的音素（避免依赖 G2P）
			String phonemes = "ㄋㄧ3 ㄏㄠ3 ㄨㄛ3 ㄕ4 Kokoro";
			TtsResult result = tts.synthesizeFromPhonemes(phonemes, "zf_001", 1.0f);

			assertNotNull(result);
			assertEquals(24000, result.sampleRate());
			assertTrue(result.audio().length > 0, "Audio should not be empty");
			assertTrue(result.duration() > 0, "Duration should be positive");

			System.out.printf("Synthesized %d samples (%.2fs) from phonemes: %s%n",
				result.audio().length, result.duration(), phonemes);
		}
	}

	@Test
	void testListVoices() throws Exception {
		Path modelPath = Path.of(MODEL_DIR, "model_dynamic.onnx");
		Path voicesDir = Path.of(MODEL_DIR, "voices");
		Path configPath = Path.of(MODEL_DIR, "config.json");

		if (!Files.exists(modelPath) || !Files.exists(configPath)) {
			System.out.println("Model files not found, skipping test");
			return;
		}

		KokoroTtsConfig config = KokoroTtsConfig.builder()
			.modelPath(modelPath.toString())
			.voicesDir(voicesDir.toString())
			.configPath(configPath.toString())
			.build();

		try (KokoroTts tts = new KokoroTts(config)) {
			var voices = tts.listVoices();
			assertFalse(voices.isEmpty(), "Should have at least one voice");
			System.out.println("Available voices: " + voices.size());
		}
	}

	@Test
	void testSaveWav() throws Exception {
		Path modelPath = Path.of(MODEL_DIR, "model_dynamic.onnx");
		Path voicesDir = Path.of(MODEL_DIR, "voices");
		Path configPath = Path.of(MODEL_DIR, "config.json");

		if (!Files.exists(modelPath) || !Files.exists(configPath)) {
			System.out.println("Model files not found, skipping test");
			return;
		}

		KokoroTtsConfig config = KokoroTtsConfig.builder()
			.modelPath(modelPath.toString())
			.voicesDir(voicesDir.toString())
			.configPath(configPath.toString())
			.build();

		Path output = Files.createTempFile("kokoro-test-", ".wav");
		try (KokoroTts tts = new KokoroTts(config)) {
			String phonemes = "ㄋㄧ3 ㄏㄠ3";
			TtsResult result = tts.synthesizeFromPhonemes(phonemes, "zf_001", 1.0f);
			tts.saveWav(result, output.toString());
			assertTrue(Files.size(output) > 44, "WAV file should have content beyond header");
			System.out.println("Saved WAV: " + output + " (" + Files.size(output) + " bytes)");
		} finally {
			Files.deleteIfExists(output);
		}
	}
}
