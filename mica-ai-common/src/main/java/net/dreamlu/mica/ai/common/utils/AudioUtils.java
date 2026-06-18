/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.utils;

import net.dreamlu.mica.ai.common.exception.MicaAiException;

import javax.sound.sampled.AudioFormat;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import javax.sound.sampled.UnsupportedAudioFileException;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 音频工具类：WAV PCM 加载、重采样。
 *
 * <p>支持 16bit/32bit float PCM，自动转单声道、重采样到 16kHz。
 */
public final class AudioUtils {

	private AudioUtils() {
		// utility class
	}

	/**
	 * 从 WAV 文件路径加载并转换为 16kHz 单声道 float32 PCM。
	 *
	 * @param wavPath WAV 文件路径
	 * @return 16kHz 单声道 float32 PCM 数据
	 */
	public static float[] loadWavAsFloat(String wavPath) {
		File file = new File(wavPath);
		if (!file.exists()) {
			throw new MicaAiException("音频文件不存在: " + wavPath);
		}
		try {
			return doLoadWav(file.toPath());
		} catch (IOException | UnsupportedAudioFileException e) {
			throw new MicaAiException("加载 WAV 文件失败: " + wavPath, e);
		}
	}

	/**
	 * 从 WAV 文件加载并转换为 16kHz 单声道 float32 PCM。
	 *
	 * @param wavPath WAV 文件路径
	 * @return 16kHz 单声道 float32 PCM 数据
	 * @throws IOException 文件不存在或格式不支持
	 */
	public static float[] loadWavAsFloat(Path wavPath) throws IOException {
		if (!Files.isRegularFile(wavPath)) {
			throw new IOException("音频文件不存在: " + wavPath);
		}
		try {
			return doLoadWav(wavPath);
		} catch (UnsupportedAudioFileException e) {
			throw new IOException("不支持的音频文件格式: " + wavPath, e);
		}
	}

	private static float[] doLoadWav(Path wavPath)
		throws IOException, UnsupportedAudioFileException {
		try (AudioInputStream ais = AudioSystem.getAudioInputStream(wavPath.toFile())) {
			AudioFormat format = ais.getFormat();
			int sampleRate = (int) format.getSampleRate();
			int channels = format.getChannels();
			int sampleSizeInBits = format.getSampleSizeInBits();

			byte[] allBytes = ais.readAllBytes();

			if (sampleSizeInBits == 16) {
				float[] audio = pcm16ToFloat(allBytes, channels);
				return ensure16kMono(audio, sampleRate);
			}

			if (sampleSizeInBits == 32
				&& format.getEncoding() == AudioFormat.Encoding.PCM_FLOAT) {
				float[] audio = pcmFloatToMono(allBytes, channels);
				return ensure16kMono(audio, sampleRate);
			}

			throw new IOException("不支持的音频格式: " + sampleSizeInBits + "bit "
				+ format.getEncoding());
		}
	}

	// ==================== 内部实现 ====================

	/**
	 * 16bit PCM 字节 → float32，自动转单声道。
	 */
	private static float[] pcm16ToFloat(byte[] bytes, int channels) {
		int numSamples = bytes.length / 2;
		ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

		short[] samples = new short[numSamples];
		for (int i = 0; i < numSamples; i++) {
			samples[i] = buf.getShort();
		}

		if (channels == 1) {
			float[] audio = new float[numSamples];
			for (int i = 0; i < numSamples; i++) {
				audio[i] = samples[i] / 32768f;
			}
			return audio;
		}

		int monoLen = numSamples / channels;
		float[] mono = new float[monoLen];
		for (int i = 0; i < monoLen; i++) {
			float sum = 0;
			for (int c = 0; c < channels; c++) {
				sum += samples[i * channels + c] / 32768f;
			}
			mono[i] = sum / channels;
		}
		return mono;
	}

	/**
	 * 32bit float PCM → 单声道。
	 */
	private static float[] pcmFloatToMono(byte[] bytes, int channels) {
		int numSamples = bytes.length / 4;
		ByteBuffer buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

		if (channels == 1) {
			float[] audio = new float[numSamples];
			for (int i = 0; i < numSamples; i++) {
				audio[i] = buf.getFloat();
			}
			return audio;
		}

		int monoLen = numSamples / channels;
		float[] mono = new float[monoLen];
		for (int i = 0; i < monoLen; i++) {
			float sum = 0;
			for (int c = 0; c < channels; c++) {
				sum += buf.getFloat();
			}
			mono[i] = sum / channels;
		}
		return mono;
	}

	/**
	 * 如果不是 16kHz，则重采样到 16kHz 单声道。
	 */
	private static float[] ensure16kMono(float[] audio, int origSr) {
		if (origSr == 16000) {
			return audio;
		}
		return resample(audio, 16000, origSr);
	}

	/**
	 * 线性重采样（Kaiser 窗 FIR 滤波器，resample_poly 近似）。
	 *
	 * @param x        输入音频
	 * @param targetSr 目标采样率
	 * @param origSr   原始采样率
	 * @return 重采样后的音频
	 */
	public static float[] resample(float[] x, int targetSr, int origSr) {
		int up = targetSr;
		int down = origSr;
		int g = gcd(up, down);
		up /= g;
		down /= g;
		if (up == down) {
			return x.clone();
		}

		int lengthOut = (int) Math.ceil((double) x.length * targetSr / origSr);

		int maxRate = Math.max(up, down);
		double fc = 1.0 / maxRate;
		int halfLen = 10 * maxRate;
		int nTaps = 2 * halfLen + 1;

		// Kaiser 窗 beta=5.0
		double[] h = new double[nTaps];
		double beta = 5.0;
		double i0Beta = besselI0(beta);
		for (int i = 0; i < nTaps; i++) {
			double t = i - halfLen;
			double sinc = (t == 0) ? 1.0 : Math.sin(Math.PI * fc * t) / (Math.PI * fc * t);
			double arg = beta * Math.sqrt(1.0 - Math.pow(2.0 * t / (nTaps - 1), 2));
			h[i] = sinc * besselI0(arg) / i0Beta * up;
		}

		double[] xUp = new double[x.length * up + nTaps];
		for (int i = 0; i < x.length; i++) {
			xUp[i * up] = x[i];
		}

		double[] yFull = new double[xUp.length + nTaps - 1];
		for (int i = 0; i < xUp.length; i++) {
			if (xUp[i] == 0) {
				continue;
			}
			for (int j = 0; j < nTaps; j++) {
				yFull[i + j] += xUp[i] * h[j];
			}
		}

		int offset = (nTaps - 1) / 2;
		float[] y = new float[lengthOut];
		for (int i = 0; i < lengthOut; i++) {
			int idx = offset + i * down;
			y[i] = (idx < yFull.length) ? (float) yFull[idx] : 0f;
		}
		return y;
	}

	private static double besselI0(double x) {
		double sum = 1.0;
		double term = 1.0;
		double xHalf = x / 2.0;
		for (int k = 1; k <= 20; k++) {
			term *= (xHalf / k) * (xHalf / k);
			sum += term;
			if (term < 1e-12 * sum) {
				break;
			}
		}
		return sum;
	}

	private static int gcd(int a, int b) {
		while (b != 0) {
			int t = b;
			b = a % b;
			a = t;
		}
		return a;
	}
}
