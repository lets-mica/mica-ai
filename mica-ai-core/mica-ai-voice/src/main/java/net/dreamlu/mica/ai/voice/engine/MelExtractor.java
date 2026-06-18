package net.dreamlu.mica.ai.voice.engine;

/**
 * 纯 Java 实现的梅尔特征提取器（对齐 torchaudio & funasr）。
 *
 * <p>对应 Python 端 {@code NumPyMelExtractor}：
 * <ul>
 *   <li>STFT + 功率谱 + 梅尔滤波 + Log Mel</li>
 *   <li>LFR (Low Frame Rate) 7 帧拼接、6 帧跳跃 → 输出 560 维特征</li>
 * </ul>
 */
public final class MelExtractor {

	private final int sr;
	private final int nFft;
	private final int nMels;
	private final int hopLength;
	private final float preEmphasis;
	private final float[] window;
	private final float[][] filters;

	/**
	 * 使用默认参数构造（对齐 SenseVoice 默认配置）。
	 */
	public MelExtractor() {
		this(16000, 400, 80, 20, 8000);
	}

	public MelExtractor(int sr, int nFft, int nMels, int fMin, int fMax) {
		this.sr = sr;
		this.nFft = nFft;
		this.nMels = nMels;
		this.hopLength = 160;
		this.preEmphasis = 0.97f;

		// 汉明窗
		this.window = new float[nFft];
		for (int i = 0; i < nFft; i++) {
			window[i] = (float) (0.54 - 0.46 * Math.cos(2.0 * Math.PI * i / nFft));
		}

		// 梅尔滤波器组
		this.filters = buildMelFilters(sr, nFft, nMels, fMin, fMax);
	}

	private static float[][] buildMelFilters(int sr, int nFft, int nMels, int fMin, int fMax) {
		int nFreqs = nFft / 2 + 1;

		// Hz <-> Mel 转换
		double[] allFreqs = new double[nFreqs];
		for (int i = 0; i < nFreqs; i++) {
			allFreqs[i] = (double) i * sr / nFft;
		}

		double melMin = 2595.0 * Math.log10(1.0 + fMin / 700.0);
		double melMax = 2595.0 * Math.log10(1.0 + fMax / 700.0);
		double[] mPts = new double[nMels + 2];
		for (int i = 0; i < nMels + 2; i++) {
			double mel = melMin + (melMax - melMin) * i / (nMels + 1);
			mPts[i] = 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
		}

		float[][] fb = new float[nFreqs][nMels];
		for (int m = 0; m < nMels; m++) {
			double fLeft = mPts[m];
			double fCenter = mPts[m + 1];
			double fRight = mPts[m + 2];
			double dLeft = fCenter - fLeft;
			double dRight = fRight - fCenter;
			for (int k = 0; k < nFreqs; k++) {
				double f = allFreqs[k];
				double wLeft = Math.max(0, (f - fLeft) / dLeft);
				double wRight = Math.max(0, (fRight - f) / dRight);
				fb[k][m] = (float) Math.min(wLeft, wRight);
			}
		}
		return fb;
	}

	/**
	 * 从音频 PCM 数据提取 LFR 特征。
	 *
	 * @param audio 16kHz 单声道 float32 音频数据
	 * @return LFR 特征矩阵 [T, 560]
	 */
	public float[][] extract(float[] audio) {
		// 1. 均值归一化
		float mean = 0;
		for (float v : audio) {
			mean += v;
		}
		mean /= audio.length;
		float[] norm = new float[audio.length];
		for (int i = 0; i < audio.length; i++) {
			norm[i] = audio[i] - mean;
		}

		// 2. 预加重
		float[] pe = new float[norm.length];
		pe[0] = norm[0];
		for (int i = 1; i < norm.length; i++) {
			pe[i] = norm[i] - preEmphasis * norm[i - 1];
		}

		// 3. STFT (加窗 → FFT → 功率谱)
		int halfNfft = nFft / 2;
		int paddedLen = pe.length + nFft; // 左右各 pad halfNfft
		float[] padded = new float[paddedLen];
		System.arraycopy(pe, 0, padded, halfNfft, pe.length);

		int numFrames = 1 + (paddedLen - nFft) / hopLength;
		int nFreqs = halfNfft + 1;

		// 功率谱 → mel spec
		float[][] melSpec = new float[numFrames][nMels];
		double[] frame = new double[nFft];
		double[] realPart = new double[nFft];
		double[] imagPart = new double[nFft];

		for (int t = 0; t < numFrames; t++) {
			int offset = t * hopLength;
			// 加窗
			for (int i = 0; i < nFft; i++) {
				frame[i] = padded[offset + i] * window[i];
			}
			// FFT
			fft(frame, realPart, imagPart, nFft);
			// 功率谱 × 梅尔滤波器
			for (int m = 0; m < nMels; m++) {
				float sum = 0;
				for (int k = 0; k < nFreqs; k++) {
					float mag2 = (float) (realPart[k] * realPart[k] + imagPart[k] * imagPart[k]);
					sum += mag2 * filters[k][m];
				}
				melSpec[t][m] = (float) Math.log(sum + 1e-7);
			}
		}

		// 4. LFR Stack (7 帧拼接, 6 帧跳跃)
		int tMel = numFrames;
		int tLfr = (tMel + 5) / 6;
		int targetLen = tLfr * 6 + 7;

		// 构造 padded mel: 左边复制 3 帧，右边补齐
		float[][] paddedMel = new float[targetLen][nMels];
		// 左填充 (重复第一帧 3 次)
		for (int i = 0; i < 3; i++) {
			System.arraycopy(melSpec[0], 0, paddedMel[i], 0, nMels);
		}
		// 原始数据
		for (int i = 0; i < tMel; i++) {
			System.arraycopy(melSpec[i], 0, paddedMel[i + 3], 0, nMels);
		}
		// 右填充 (重复最后一帧)
		for (int i = 3 + tMel; i < targetLen; i++) {
			System.arraycopy(melSpec[tMel - 1], 0, paddedMel[i], 0, nMels);
		}

		// LFR 拼接: 每帧取 7 个间隔 6 帧的 80 维拼接为 560 维
		float[][] lfrFeat = new float[tLfr][560];
		for (int i = 0; i < tLfr; i++) {
			for (int j = 0; j < 7; j++) {
				int srcFrame = j + i * 6;
				System.arraycopy(paddedMel[srcFrame], 0, lfrFeat[i], j * 80, 80);
			}
		}
		return lfrFeat;
	}

	/**
	 * 通用 FFT（支持任意大小，Bluestein 算法 + 基 2 Cooley-Tukey）。
	 */
	private static void fft(double[] input, double[] real, double[] imag, int n) {
		if (n <= 0) return;
		if (n == 1) {
			real[0] = input[0];
			imag[0] = 0;
			return;
		}

		// 如果是 2 的幂，直接用基 2 FFT
		if ((n & (n - 1)) == 0) {
			fftRadix2(input, real, imag, n);
			return;
		}

		// Bluestein 算法：将 N 点 DFT 转化为循环卷积，用 2 的幂 FFT 计算
		int m = Integer.highestOneBit(2 * n - 1) << 1; // >= 2N-1 的最小 2 的幂

		double[] chirpReal = new double[n];
		double[] chirpImag = new double[n];
		for (int k = 0; k < n; k++) {
			double angle = Math.PI * (long) k * k / n;
			chirpReal[k] = Math.cos(angle);
			chirpImag[k] = -Math.sin(angle);
		}

		// a[n] = input[n] * conj(chirp[n])
		double[] aReal = new double[m];
		double[] aImag = new double[m];
		for (int i = 0; i < n; i++) {
			aReal[i] = input[i] * chirpReal[i];
			aImag[i] = -input[i] * chirpImag[i]; // input is real, so imag=0, result = input * conj(chirp)
		}

		// b[n] = chirp[n] (with wrap-around for negative indices)
		double[] bReal = new double[m];
		double[] bImag = new double[m];
		bReal[0] = chirpReal[0];
		bImag[0] = chirpImag[0];
		for (int i = 1; i < n; i++) {
			bReal[i] = chirpReal[i];
			bImag[i] = chirpImag[i];
			bReal[m - i] = chirpReal[i];
			bImag[m - i] = chirpImag[i];
		}

		// FFT(a) and FFT(b)
		double[] faReal = new double[m];
		double[] faImag = new double[m];
		double[] fbReal = new double[m];
		double[] fbImag = new double[m];
		fftRadix2(aReal, faReal, faImag, m);
		fftRadix2(bReal, fbReal, fbImag, m);

		// Pointwise multiply: FA * FB
		double[] fcReal = new double[m];
		double[] fcImag = new double[m];
		for (int i = 0; i < m; i++) {
			fcReal[i] = faReal[i] * fbReal[i] - faImag[i] * fbImag[i];
			fcImag[i] = faReal[i] * fbImag[i] + faImag[i] * fbReal[i];
		}

		// IFFT(fc) via conjugate trick: IFFT(x) = conj(FFT(conj(x))) / m
		double[] ifftInReal = new double[m];
		double[] ifftInImag = new double[m];
		for (int i = 0; i < m; i++) {
			ifftInReal[i] = fcReal[i];
			ifftInImag[i] = -fcImag[i];
		}
		double[] convReal = new double[m];
		double[] convImag = new double[m];
		fftRadix2(ifftInReal, convReal, convImag, m);

		// Extract result: X[k] = chirp[k] * (conv[k] / m)
		for (int k = 0; k < n; k++) {
			double cr = convReal[k] / m;
			double ci = -convImag[k] / m; // undo conjugate
			// X[k] = chirp[k] * (cr + j*ci)
			real[k] = chirpReal[k] * cr - chirpImag[k] * ci;
			imag[k] = chirpReal[k] * ci + chirpImag[k] * cr;
		}
	}

	/**
	 * Cooley-Tukey 基 2 FFT（就地计算），n 必须为 2 的幂。
	 */
	private static void fftRadix2(double[] input, double[] real, double[] imag, int n) {
		// 位反转排列
		int bits = Integer.numberOfTrailingZeros(n);
		for (int i = 0; i < n; i++) {
			int j = Integer.reverse(i) >>> (32 - bits);
			real[i] = (j < n) ? input[j] : 0;
			imag[i] = 0;
		}

		// 蝶形运算
		for (int len = 2; len <= n; len <<= 1) {
			double angle = -2.0 * Math.PI / len;
			double wReal = Math.cos(angle);
			double wImag = Math.sin(angle);
			for (int i = 0; i < n; i += len) {
				double curReal = 1.0, curImag = 0.0;
				for (int j = 0; j < len / 2; j++) {
					double tReal = curReal * real[i + j + len / 2] - curImag * imag[i + j + len / 2];
					double tImag = curReal * imag[i + j + len / 2] + curImag * real[i + j + len / 2];
					real[i + j + len / 2] = real[i + j] - tReal;
					imag[i + j + len / 2] = imag[i + j] - tImag;
					real[i + j] += tReal;
					imag[i + j] += tImag;
					double newCurReal = curReal * wReal - curImag * wImag;
					curImag = curReal * wImag + curImag * wReal;
					curReal = newCurReal;
				}
			}
		}
	}

	public int getSr() {
		return sr;
	}

	public int getNFft() {
		return nFft;
	}

	public int getNMels() {
		return nMels;
	}
}

