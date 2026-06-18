package net.dreamlu.mica.ai.tts;

import net.dreamlu.mica.ai.tts.g2p.ChineseG2P;
import net.dreamlu.mica.ai.tts.internal.TextFrontend;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 文本前端测试。
 */
class TextFrontendTest {

	@Test
	void testSplitPhonemesSimple() {
		String phonemes = "ni3 hao3 wo3 shi4 Kokoro";
		List<String> batches = TextFrontend.splitPhonemes(phonemes);
		assertFalse(batches.isEmpty());
	}

	@Test
	void testSplitPhonemesWithPunctuation() {
		String phonemes = "ni3 hao3. wo3 shi4 Kokoro. zai4 jian4!";
		List<String> batches = TextFrontend.splitPhonemes(phonemes);
		assertTrue(batches.size() >= 1);
	}

	@Test
	void testSplitPhonemesEmpty() {
		List<String> batches = TextFrontend.splitPhonemes("");
		assertTrue(batches.isEmpty());
	}

	@Test
	void testTrimSilence() {
		float[] audio = new float[1000];
		// 在中间放一些非零值
		for (int i = 100; i < 500; i++) {
			audio[i] = (float) Math.sin(i * 0.1);
		}
		float[] trimmed = TextFrontend.trimSilence(audio, 0.01f);
		assertTrue(trimmed.length < audio.length);
		assertTrue(trimmed.length > 0);
	}

	@Test
	void testTrimSilenceAllSilent() {
		float[] audio = new float[100];
		float[] trimmed = TextFrontend.trimSilence(audio, 0.01f);
		assertEquals(0, trimmed.length);
	}

	@Test
	void testChineseG2PBasic() {
		String pinyin = "ni3";
		String bpmf = ChineseG2P.pinyinToBopomofo(pinyin);
		assertEquals("ㄋㄧ", bpmf);
	}

	@Test
	void testChineseG2PConvert() {
		String result = ChineseG2P.getDefault().convert("你好");
		assertNotNull(result);
		assertTrue(result.contains("ㄋ") || result.contains("ㄏ"));
	}
}
