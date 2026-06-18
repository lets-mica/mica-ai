/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.speaker.internal;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import javax.sound.sampled.AudioFileFormat;
import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * AudioUtils 测试：WAV 加载、PCM 转换、重采样。
 */
@DisplayName("AudioUtils")
class AudioUtilsTest {

	// ==================== 异常路径 ====================

	@Test
	@DisplayName("文件不存在时抛出 IOException")
	void testLoadNonexistentFile() {
		Path nonexistent = Path.of("nonexistent_audio_2026.wav");
		IOException ex = assertThrows(IOException.class,
			() -> AudioUtils.loadWavAsFloat(nonexistent));
		assertTrue(ex.getMessage().contains("不存在"));
	}

	// ==================== 16bit 单声道 16kHz WAV ====================

	@Test
	@DisplayName("16bit 16kHz 单声道 WAV 加载回环")
	void test16kMono16bitRoundtrip() throws Exception {
		// 生成 440Hz 正弦波测试数据
		float[] original = generateSine(440, 1.0, 16000, 0.5f);
		Path tempWav = writePcm16MonoWav(original, 16000);

		try {
			float[] loaded = AudioUtils.loadWavAsFloat(tempWav);

			// 信号完整性：非空、采样点数合理
			assertNotNull(loaded);
			assertTrue(loaded.length > 0, "加载的音频不应为空");
			assertEquals(original.length, loaded.length, 10,
				"加载后采样点数应与原始接近");

			// 精度验证（PCM 16bit 有 1/32768 ≈ 3e-5 的量化误差）
			for (int i = 0; i < Math.min(100, loaded.length); i++) {
				assertEquals(original[i], loaded[i], 0.001,
					"第 " + i + " 个采样点偏差过大");
			}
		} finally {
			Files.deleteIfExists(tempWav);
		}
	}

	@Test
	@DisplayName("加载全零信号")
	void testLoadSilence() throws Exception {
		float[] original = new float[8000]; // 0.5s 静音
		Path tempWav = writePcm16MonoWav(original, 16000);

		try {
			float[] loaded = AudioUtils.loadWavAsFloat(tempWav);
			assertEquals(8000, loaded.length);
			for (int i = 0; i < loaded.length; i++) {
				assertEquals(0f, loaded[i], 0.001);
			}
		} finally {
			Files.deleteIfExists(tempWav);
		}
	}

	// ==================== 16bit 双声道 → 单声道 ====================

	@Test
	@DisplayName("16bit 双声道 16kHz → 单声道取平均")
	void testStereoToMono16bit() throws Exception {
		// 左声道 = 0.5, 右声道 = -0.5 → 平均 = 0
		int sr = 16000;
		int n = 8000;
		short[] stereo = new short[n * 2];
		for (int i = 0; i < n; i++) {
			stereo[i * 2] = (short) (0.5 * 32767);      // 左
			stereo[i * 2 + 1] = (short) (-0.5 * 32767); // 右
		}

		Path tempWav = writePcm16Wav(stereo, sr, 2);

		try {
			float[] mono = AudioUtils.loadWavAsFloat(tempWav);
			assertEquals(n, mono.length);
			// 平均应接近 0
			for (int i = 0; i < Math.min(100, mono.length); i++) {
				assertEquals(0f, mono[i], 0.01);
			}
		} finally {
			Files.deleteIfExists(tempWav);
		}
	}

	// ==================== 重采样 ====================

	@Nested
	@DisplayName("重采样")
	class Resample {

		@Test
		@DisplayName("相同采样率返回 clone（非同一引用）")
		void testResampleSameRate() {
			float[] x = {1f, 2f, 3f, 4f, 5f};
			float[] y = AudioUtils.resample(x, 16000, 16000);
			assertArrayEquals(x, y, 0.001f);
			assertNotSame(x, y, "相同采样率应返回 clone");
		}

		@Test
		@DisplayName("48kHz → 16kHz 降采样")
		void testResampleDown() {
			// 生成 48kHz 正弦波 1s
			float[] x = generateSine(440, 1.0, 48000, 0.5f);
			float[] y = AudioUtils.resample(x, 16000, 48000);

			// 输出长度应约为输入长度 × 16000/48000
			int expectedLen = (int) Math.ceil((double) x.length * 16000 / 48000);
			assertEquals(expectedLen, y.length, 2,
				"降采样长度: 预期=" + expectedLen + " 实际=" + y.length);

			// 无 NaN / Inf
			for (float v : y) {
				assertFalse(Float.isNaN(v));
				assertFalse(Float.isInfinite(v));
			}
		}

		@Test
		@DisplayName("8kHz → 16kHz 升采样")
		void testResampleUp() {
			float[] x = generateSine(440, 0.5, 8000, 0.5f);
			float[] y = AudioUtils.resample(x, 16000, 8000);

			int expectedLen = (int) Math.ceil((double) x.length * 16000 / 8000);
			assertEquals(expectedLen, y.length, 2);

			for (float v : y) {
				assertFalse(Float.isNaN(v));
				assertFalse(Float.isInfinite(v));
			}
		}

		@Test
		@DisplayName("整数倍上采样（up/down 约分后 =1）")
		void testIntegerUpSample() {
			float[] x = {1f, 2f, 3f};
			float[] y = AudioUtils.resample(x, 32000, 16000);
			assertEquals(6, y.length, 1);
		}
	}

	// ==================== 辅助方法 ====================

	/**
	 * 生成正弦波 float32 PCM（16kHz 单声道）。
	 */
	static float[] generateSine(double freqHz, double durationSec, int sampleRate, double amplitude) {
		int n = (int) (durationSec * sampleRate);
		float[] wave = new float[n];
		for (int i = 0; i < n; i++) {
			double t = (double) i / sampleRate;
			wave[i] = (float) (amplitude * Math.sin(2.0 * Math.PI * freqHz * t));
		}
		return wave;
	}

	/**
	 * 将 float32 PCM 写为 16bit 单声道 WAV 临时文件。
	 */
	static Path writePcm16MonoWav(float[] pcm, int sampleRate) throws IOException {
		short[] shorts = new short[pcm.length];
		for (int i = 0; i < pcm.length; i++) {
			int sample = Math.round(pcm[i] * 32767);
			shorts[i] = (short) Math.max(Short.MIN_VALUE, Math.min(Short.MAX_VALUE, sample));
		}
		return writePcm16Wav(shorts, sampleRate, 1);
	}

	/**
	 * 将 short PCM 写入 WAV 文件。
	 */
	static Path writePcm16Wav(short[] pcm, int sampleRate, int channels) throws IOException {
		byte[] audioBytes = new byte[pcm.length * 2];
		ByteBuffer buf = ByteBuffer.wrap(audioBytes).order(ByteOrder.LITTLE_ENDIAN);
		for (short s : pcm) {
			buf.putShort(s);
		}

		AudioFormat format = new AudioFormat(sampleRate, 16, channels, true, false);
		ByteArrayInputStream bais = new ByteArrayInputStream(audioBytes);
		AudioInputStream ais = new AudioInputStream(bais, format, pcm.length / channels);

		Path temp = Files.createTempFile("mica_test_", ".wav");
		AudioSystem.write(ais, AudioFileFormat.Type.WAVE, temp.toFile());
		return temp;
	}
}
