package net.dreamlu.mica.ai.tts;

import net.dreamlu.mica.ai.tts.config.KokoroTtsConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * KokoroTtsConfig 测试。
 */
class KokoroTtsConfigTest {

	@Test
	void testBuilder() {
		KokoroTtsConfig config = KokoroTtsConfig.builder()
			.modelPath("model.onnx")
			.voicesDir("voices")
			.configPath("config.json")
			.defaultVoice("zf_001")
			.defaultSpeed(1.0f)
			.build();

		assertEquals("model.onnx", config.getModelPath());
		assertEquals("voices", config.getVoicesDir());
		assertEquals("config.json", config.getConfigPath());
		assertEquals("zf_001", config.getDefaultVoice());
		assertEquals(1.0f, config.getDefaultSpeed());
	}

	@Test
	void testBuilderDefaults() {
		KokoroTtsConfig config = KokoroTtsConfig.builder()
			.modelPath("m")
			.voicesDir("v")
			.configPath("c")
			.build();

		assertEquals("zf_001", config.getDefaultVoice());
		assertEquals(1.0f, config.getDefaultSpeed());
		assertEquals("cpu", config.getOnnxProvider());
		assertEquals(510, KokoroTtsConfig.MAX_PHONEME_LENGTH);
		assertEquals(24000, KokoroTtsConfig.SAMPLE_RATE);
	}

	@Test
	void testBuilderMissingModelPath() {
		assertThrows(IllegalArgumentException.class, () ->
			KokoroTtsConfig.builder()
				.voicesDir("v")
				.configPath("c")
				.build()
		);
	}
}
