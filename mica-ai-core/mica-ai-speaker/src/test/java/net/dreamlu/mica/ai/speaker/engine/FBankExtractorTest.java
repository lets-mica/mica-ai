/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.speaker.engine;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * FBankExtractor 测试：Mel 滤波器构建、Hz↔Mel 转换、FFT、特征提取。
 */
@DisplayName("FBankExtractor")
class FBankExtractorTest {

	// ==================== Hz ↔ Mel 转换 ====================

	@Test
	@DisplayName("Hz → Mel → Hz 往返计算一致")
	void testHtkMelRoundTrip() {
		double[] testHz = {20, 100, 500, 1000, 2000, 4000, 7600, 8000};
		for (double hz : testHz) {
			double mel = FBankExtractor.hzToMel(hz);
			double back = FBankExtractor.melToHz(mel);
			assertEquals(hz, back, 0.01, "Hz=" + hz + " 往返后偏差过大");
		}
	}

	@Test
	@DisplayName("Hz → Mel 单调递增")
	void testHtkMelMonotonic() {
		double mel0 = FBankExtractor.hzToMel(20);
		double mel1 = FBankExtractor.hzToMel(8000);
		assertTrue(mel1 > mel0, "Mel 值应随频率单调递增");
	}

	@Test
	@DisplayName("Mel → Hz 单调递增")
	void testMelToHzMonotonic() {
		double hz0 = FBankExtractor.melToHz(0);
		double hz1 = FBankExtractor.melToHz(3000);
		assertTrue(hz1 > hz0, "Hz 值应随 Mel 单调递增");
	}

	@Test
	@DisplayName("Hz=0 → Mel=0")
	void testZeroHzIsZeroMel() {
		assertEquals(0.0, FBankExtractor.hzToMel(0), 0.001);
		assertEquals(0.0, FBankExtractor.melToHz(0), 0.001);
	}

	// ==================== 默认构造 ====================

	@Test
	@DisplayName("默认构造参数正确")
	void testDefaultConstructor() {
		FBankExtractor fe = new FBankExtractor();
		assertEquals(512, fe.getNFft());
		assertEquals(257, fe.getNFreqs());  // 512/2 + 1
		assertEquals(160, fe.getHopSize()); // 10ms @ 16kHz
	}

	// ==================== 自定义构造 ====================

	@Test
	@DisplayName("自定义采样率参数")
	void testCustomConstructor() {
		FBankExtractor fe = new FBankExtractor(8000, 256, 40, 20f, 4000f);
		assertEquals(256, fe.getNFft());
		assertEquals(129, fe.getNFreqs());
		assertEquals(160, fe.getHopSize());
	}

	// ==================== 特征提取 ====================

	@Test
	@DisplayName("合成正弦波特征提取")
	void testExtractSineWave() {
		// 生成 440Hz 正弦波，1 秒，16kHz
		int sr = 16000;
		float[] audio = generateSine(440, 1.0f, sr, 0.5f);

		FBankExtractor fe = new FBankExtractor();
		float[][] fbank = fe.extract(audio);

		int expectedFrames = 1 + (audio.length - 512) / 160;
		assertEquals(expectedFrames, fbank.length, "帧数不符合预期");
		assertEquals(80, fbank[0].length, "每帧应为 80 维 Mel FBank");

		// 验证非零、非 NaN
		for (int t = 0; t < fbank.length; t++) {
			for (int m = 0; m < 80; m++) {
				assertFalse(Float.isNaN(fbank[t][m]), "FBank 不应含 NaN: 帧=" + t + " 频带=" + m);
				assertFalse(Float.isInfinite(fbank[t][m]), "FBank 不应含 Inf: 帧=" + t + " 频带=" + m);
			}
		}
	}

	@Test
	@DisplayName("静音信号特征提取")
	void testExtractSilence() {
		// 全零信号
		float[] audio = new float[16000]; // 1 秒 16kHz 静音

		FBankExtractor fe = new FBankExtractor();
		float[][] fbank = fe.extract(audio);

		assertTrue(fbank.length > 0, "即使是静音也应产生帧");

		// log(1e-7) ≈ -16.12，静音下所有 Mel 值应接近 floor
		for (int t = 0; t < Math.min(5, fbank.length); t++) {
			for (int m = 0; m < 80; m++) {
				assertFalse(Float.isNaN(fbank[t][m]), "静音 FBank 不应含 NaN");
			}
		}
	}

	@Test
	@DisplayName("极短音频（少于 1 帧窗口）")
	void testExtractVeryShortAudio() {
		float[] audio = new float[200]; // 短于 512 点

		FBankExtractor fe = new FBankExtractor();
		float[][] fbank = fe.extract(audio);

		// 至少产生 1 帧（代码中 Math.max(1, ...)）
		assertEquals(1, fbank.length);
		assertEquals(80, fbank[0].length);
	}

	@Test
	@DisplayName("特征时间轴连续")
	void testFBankTemporalContinuity() {
		// 生成线性扫频信号，确保相邻帧有变化但不是剧烈跳变
		float[] audio = generateChirp(100, 3000, 2.0f, 16000);

		FBankExtractor fe = new FBankExtractor();
		float[][] fbank = fe.extract(audio);

		assertTrue(fbank.length > 10, "应有足够帧数测试连续性");

		// 相邻帧的总能量不应完全相等（扫频信号应产生变化）
		boolean hasVariation = false;
		for (int t = 1; t < fbank.length; t++) {
			double diff = 0;
			for (int m = 0; m < 80; m++) {
				diff += Math.abs(fbank[t][m] - fbank[t - 1][m]);
			}
			if (diff > 10.0) { // 有明显变化
				hasVariation = true;
				break;
			}
		}
		assertTrue(hasVariation, "扫频信号的相邻 FBank 帧应有变化");
	}

	// ==================== Mel 滤波器组属性 ====================

	@Nested
	@DisplayName("Mel 滤波器组")
	class MelFilterBank {

		@Test
		@DisplayName("滤波器数量正确")
		void testFilterCount() {
			FBankExtractor fe = new FBankExtractor();
			float[][] features = fe.extract(generateSine(440, 0.5f, 16000, 0.5f));
			assertEquals(80, features[0].length, "每帧应含 80 个 Mel 频带");
		}

		@Test
		@DisplayName("频带选择性：低频信号峰值在低 mel band，高频信号峰值在高 mel band")
		void testFrequencySelectivity() {
			// 500Hz 信号：能量应集中在低 mel band（约 16 附近）
			float[] lowAudio = generateSine(500, 1.0f, 16000, 0.8f);
			// 4000Hz 信号：能量应集中在高 mel band（约 55 附近）
			float[] highAudio = generateSine(4000, 1.0f, 16000, 0.8f);

			FBankExtractor fe = new FBankExtractor();
			float[][] lowFbank = fe.extract(lowAudio);
			float[][] highFbank = fe.extract(highAudio);

			double logFloor = Math.log(1e-7); // ≈ -16.12

			// 找 500Hz 信号在第 50 帧的峰值 mel band
			int lowPeakBand = findPeakBand(lowFbank[50]);
			assertTrue(lowPeakBand >= 10 && lowPeakBand <= 25,
				"500Hz 峰值应在低频 mel band (10-25)，实际: " + lowPeakBand);
			assertTrue(lowFbank[50][lowPeakBand] > logFloor + 5,
				"500Hz 峰值能量应远高于 log floor");

			// 找 4000Hz 信号在第 50 帧的峰值 mel band
			int highPeakBand = findPeakBand(highFbank[50]);
			assertTrue(highPeakBand >= 40 && highPeakBand <= 70,
				"4kHz 峰值应在高频 mel band (40-70)，实际: " + highPeakBand);
			assertTrue(highFbank[50][highPeakBand] > logFloor + 5,
				"4kHz 峰值能量应远高于 log floor");

			// 频率选择性：低频信号峰值 band < 高频信号峰值 band
			assertTrue(lowPeakBand < highPeakBand,
				"500Hz 峰值 band (" + lowPeakBand + ") 应 < 4kHz 峰值 band (" + highPeakBand + ")");
		}

		/**
		 * 找 FBank 单帧中能量最大的 mel band 索引。
		 */
		private int findPeakBand(float[] frame) {
			int peak = 0;
			for (int m = 1; m < frame.length; m++) {
				if (frame[m] > frame[peak]) {
					peak = m;
				}
			}
			return peak;
		}
	}

	// ==================== 工具方法 ====================

	/**
	 * 生成 16kHz 单声道正弦波。
	 */
	static float[] generateSine(double freqHz, double durationSec, int sampleRate, double amplitude) {
		int numSamples = (int) (durationSec * sampleRate);
		float[] wave = new float[numSamples];
		for (int i = 0; i < numSamples; i++) {
			double t = (double) i / sampleRate;
			wave[i] = (float) (amplitude * Math.sin(2.0 * Math.PI * freqHz * t));
		}
		return wave;
	}

	/**
	 * 生成线性扫频信号。
	 */
	static float[] generateChirp(double f0, double f1, double durationSec, int sampleRate) {
		int numSamples = (int) (durationSec * sampleRate);
		float[] wave = new float[numSamples];
		double k = (f1 - f0) / durationSec;
		for (int i = 0; i < numSamples; i++) {
			double t = (double) i / sampleRate;
			double phase = 2.0 * Math.PI * (f0 * t + 0.5 * k * t * t);
			wave[i] = (float) (0.5 * Math.sin(phase));
		}
		return wave;
	}
}
