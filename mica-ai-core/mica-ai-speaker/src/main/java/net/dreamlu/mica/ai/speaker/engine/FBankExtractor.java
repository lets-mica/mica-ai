/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.speaker.engine;

import org.jtransforms.fft.DoubleFFT_1D;

/**
 * 80 维 log-Mel FBank 特征提取器（ERes2Net 前端，HTK 标准 Mel 滤波器组）。
 *
 * <p>参数对齐 3D-Speaker ERes2Net-base 16k 模型：
 * <ul>
 *   <li>采样率：16000 Hz</li>
 *   <li>窗长：25ms (400 samples)，帧移：10ms (160 samples)</li>
 *   <li>FFT 点数：512（JTransforms 实现）</li>
 *   <li>频率范围：20 ~ 7600 Hz</li>
 *   <li>Mel 频带数：80（HTK 标准三角滤波器）</li>
 * </ul>
 *
 * <p>输出：{@code float[T][80]}，每行为一帧的 80 维 log-Mel FBank 特征。
 */
public final class FBankExtractor {

	private static final int SR = 16000;
	private static final int N_FFT = 512;
	private static final int N_MELS = 80;
	private static final int WINDOW_SIZE = 400; // 25ms
	private static final int HOP_SIZE = 160;    // 10ms
	private static final float F_MIN = 20f;
	private static final float F_MAX = 7600f;
	private static final float PRE_EMPHASIS = 0.97f;
	private static final float LOG_FLOOR = 1e-7f;

	private final int nFft;
	private final int nFreqs;
	private final int hopSize;
	private final float preEmphasis;
	private final float[] window;
	private final float[][] filters;
	private final DoubleFFT_1D fft;

	/**
	 * 使用默认参数构造（16kHz / 512 FFT / 80 Mel / 20-7600 Hz）。
	 */
	public FBankExtractor() {
		this(SR, N_FFT, N_MELS, F_MIN, F_MAX);
	}

	/**
	 * 自定义参数构造。
	 *
	 * @param sr     采样率 (Hz)
	 * @param nFft   FFT 点数（必须为 2 的幂）
	 * @param nMels  Mel 频带数
	 * @param fMin   最低频率 (Hz)
	 * @param fMax   最高频率 (Hz)
	 */
	public FBankExtractor(int sr, int nFft, int nMels, float fMin, float fMax) {
		this.nFft = nFft;
		this.nFreqs = nFft / 2 + 1;
		this.hopSize = HOP_SIZE;
		this.preEmphasis = PRE_EMPHASIS;

		// 汉明窗
		this.window = new float[nFft];
		for (int i = 0; i < nFft; i++) {
			window[i] = (float) (0.54 - 0.46 * Math.cos(2.0 * Math.PI * i / nFft));
		}

		// HTK 标准 Mel 三角滤波器组
		this.filters = buildHtkMelFilters(sr, nFft, nMels, fMin, fMax);

		// JTransforms FFT
		this.fft = new DoubleFFT_1D(nFft);
	}

	/**
	 * 从 16kHz 单声道 PCM 提取 log-Mel FBank 特征。
	 *
	 * @param audio 16kHz 单声道 float32 PCM 数据
	 * @return 特征矩阵 [T][80]，T 为帧数
	 */
	public float[][] extract(float[] audio) {
		// 1. 直流分量去除（减均值）
		double sum = 0;
		for (float v : audio) {
			sum += v;
		}
		float mean = (float) (sum / audio.length);
		float[] dcRemoved = new float[audio.length];
		for (int i = 0; i < audio.length; i++) {
			dcRemoved[i] = audio[i] - mean;
		}

		// 2. 预加重
		float[] pe = new float[dcRemoved.length];
		pe[0] = dcRemoved[0];
		for (int i = 1; i < dcRemoved.length; i++) {
			pe[i] = dcRemoved[i] - preEmphasis * dcRemoved[i - 1];
		}

		// 3. 分帧 + 加窗 + FFT → 功率谱 → Mel 滤波 → log
		int numFrames = Math.max(1, 1 + (pe.length - nFft) / hopSize);
		float[][] fbank = new float[numFrames][N_MELS];

		// JTransforms realForward 格式：[r0, r1, r2, ..., r_{n/2}, i_{n/2-1}, ..., i1]
		// 即前 n/2+1 为实部（正频率），后 n/2-1 为虚部（负频率倒序）
		double[] fftBuf = new double[nFft];
		int halfNfft = nFft / 2;

		for (int t = 0; t < numFrames; t++) {
			int offset = t * hopSize;

			// 加汉明窗，填入 FFT 缓冲区
			for (int i = 0; i < nFft; i++) {
				fftBuf[i] = (i + offset < pe.length) ? (pe[i + offset] * window[i]) : 0.0;
			}

			// JTransforms realForward：就地计算
			fft.realForward(fftBuf);

			// 功率谱 × Mel 滤波器组
			// JTransforms realForward packed 格式（n 为偶数）：
			//   fftBuf[0]   = Re[0]      (DC)
			//   fftBuf[1]   = Re[n/2]    (Nyquist)
			//   fftBuf[2*k] = Re[k],  fftBuf[2*k+1] = Im[k],  k=1..n/2-1
			for (int m = 0; m < N_MELS; m++) {
				float melSum = 0f;
				for (int k = 0; k < nFreqs; k++) {
					double re, im;
					if (k == 0) {
						re = fftBuf[0];          // DC
						im = 0;
					} else if (k == halfNfft) {
						re = fftBuf[1];          // Nyquist
						im = 0;
					} else {
						re = fftBuf[2 * k];      // Re[k]
						im = fftBuf[2 * k + 1];  // Im[k]
					}
					float power = (float) (re * re + im * im);
					melSum += power * filters[k][m];
				}
				fbank[t][m] = (float) Math.log(Math.max(melSum, LOG_FLOOR));
			}
		}

		return fbank;
	}

	/**
	 * 构建 HTK 标准 Mel 三角滤波器组。
	 *
	 * <p>流程：
	 * <ol>
	 *   <li>Hz → Mel：mel = 2595 * log10(1 + f/700)</li>
	 *   <li>Mel 域均匀取 nMels+2 个点（包含边界）</li>
	 *   <li>Mel → Hz：f = 700 * (10^(mel/2595) - 1)</li>
	 *   <li>Hz → FFT bin 索引</li>
	 *   <li>三角窗权重归一化（面积 = 1）</li>
	 * </ol>
	 */
	private static float[][] buildHtkMelFilters(int sr, int nFft, int nMels,
												 float fMin, float fMax) {
		int nFreqs = nFft / 2 + 1;

		// 各 FFT bin 对应的实际频率
		double[] binFreqs = new double[nFreqs];
		for (int i = 0; i < nFreqs; i++) {
			binFreqs[i] = (double) i * sr / nFft;
		}

		// Hz → Mel 转换（HTK 公式）
		double melMin = hzToMel(fMin);
		double melMax = hzToMel(fMax);

		// Mel 域均匀采样 nMels+2 个点
		double[] melPts = new double[nMels + 2];
		for (int i = 0; i < nMels + 2; i++) {
			double mel = melMin + (melMax - melMin) * i / (nMels + 1);
			melPts[i] = melToHz(mel);
		}

		// 构造三角滤波器矩阵 [nFreqs][nMels]
		float[][] fb = new float[nFreqs][nMels];

		for (int m = 0; m < nMels; m++) {
			double fLeft = melPts[m];
			double fCenter = melPts[m + 1];
			double fRight = melPts[m + 2];

			for (int k = 0; k < nFreqs; k++) {
				double f = binFreqs[k];
				if (f <= fLeft || f >= fRight) {
					fb[k][m] = 0f;
				} else if (f <= fCenter) {
					// 上升沿
					fb[k][m] = (float) ((f - fLeft) / (fCenter - fLeft));
				} else {
					// 下降沿
					fb[k][m] = (float) ((fRight - f) / (fRight - fCenter));
				}
			}
		}

		// 面积归一化：每个 filter 的权重和归一化
		for (int m = 0; m < nMels; m++) {
			float weightSum = 0f;
			for (int k = 0; k < nFreqs; k++) {
				weightSum += fb[k][m];
			}
			if (weightSum > 0) {
				for (int k = 0; k < nFreqs; k++) {
					fb[k][m] /= weightSum;
				}
			}
		}

		return fb;
	}

	// ==================== Hz ↔ Mel 转换（HTK 标准） ====================

	/**
	 * HTK 标准 Hz → Mel。
	 */
	static double hzToMel(double hz) {
		return 2595.0 * Math.log10(1.0 + hz / 700.0);
	}

	/**
	 * HTK 标准 Mel → Hz。
	 */
	static double melToHz(double mel) {
		return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
	}

	// ==================== Getters ====================

	public int getNFft() {
		return nFft;
	}

	public int getNFreqs() {
		return nFreqs;
	}

	public int getHopSize() {
		return hopSize;
	}
}
