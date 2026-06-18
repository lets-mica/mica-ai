package net.dreamlu.mica.ai.voice.engine;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.voice.config.*;

import javax.sound.sampled.*;
import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * SenseVoice ONNX 语音识别推理引擎（Java 17 移植版）。
 *
 * <p>零 PyTorch 依赖，纯 ONNX Runtime 推理，支持：
 * <ul>
 *   <li>动态 Prompt 构造（语言 ID + ITN 开关）</li>
 *   <li>Trie 树加速热词召回</li>
 *   <li>自动分段拼接（长音频支持）</li>
 *   <li>中文 ITN（数字规范化）</li>
 * </ul>
 *
 * <p>典型用法：
 * <pre>{@code
 * SenseVoiceConfig config = new SenseVoiceConfig()
 *     .encoderPath("models/encoder.onnx")
 *     .decoderPath("models/decoder.onnx")
 *     .tokenizerPath("models/tokenizer.model")
 *     .hotWords(List.of("mica", "梦想卢"));
 *
 * try (SenseVoice voice = new SenseVoice(config)) {
 *     TranscriptionResult result = voice.recognize(audioData);
 *     System.out.println(result.text());
 * }
 * }</pre>
 */
@Slf4j
public final class SenseVoice implements Closeable, AutoCloseable {

	private final SenseVoiceConfig config;
	private final OrtEnvironment env;
	private final SenseVoiceEncoder encoder;
	private final SenseVoiceDecoder decoder;
	private final MelExtractor frontend;
	private final SentencePieceTokenizer tokenizer;
	private final HotwordRadar radar;
	private boolean closed = false;

	public SenseVoice(SenseVoiceConfig config) {
		this.config = config;
		requireFile(config.getEncoderPath(), "encoderPath");
		requireFile(config.getDecoderPath(), "decoderPath");
		requireFile(config.getTokenizerPath(), "tokenizerPath");

		try {
			this.env = OrtEnvironment.getEnvironment();
			this.encoder = new SenseVoiceEncoder(config.getEncoderPath(), env);
			this.decoder = new SenseVoiceDecoder(config.getDecoderPath(), env);
			this.frontend = new MelExtractor();
			this.tokenizer = new SentencePieceTokenizer(config.getTokenizerPath());
			this.radar = new HotwordRadar(
				config.getHotwords() != null ? config.getHotwords() : List.of(),
				tokenizer
			);
		} catch (OrtException | IOException e) {
			throw new RuntimeException("初始化 SenseVoice 引擎失败: " + e.getMessage(), e);
		}
		log.info("SenseVoice 引擎初始化完成");
	}

	/**
	 * 更新热词列表。
	 *
	 * @param hotwords 新的热词列表
	 */
	public void updateHotwords(List<String> hotwords) {
		requireOpen();
		radar.updateHotwords(hotwords);
	}

	/**
	 * 识别 float32 PCM 音频数据（16kHz 单声道）。
	 *
	 * @param audioData 16kHz 单声道 float32 PCM 音频数据
	 * @return 转录结果
	 */
	public TranscriptionResult recognize(float[] audioData) {
		return recognize(audioData, "auto", config.isItn(), 40, 5);
	}

	/**
	 * 识别 float32 PCM 音频数据（16kHz 单声道）。
	 *
	 * @param audioData 16kHz 单声道 float32 PCM 音频数据
	 * @param lid       语言 ID (如 "auto", "zh", "en")
	 * @param itn       是否启用 ITN
	 * @param chunkSize 分段大小（秒）
	 * @param overlap   分段重叠（秒）
	 * @return 转录结果
	 */
	public TranscriptionResult recognize(float[] audioData, String lid, boolean itn,
										 int chunkSize, int overlap) {
		requireOpen();

		long tStart = System.nanoTime();

		// 1. 提取全量特征
		long t0 = System.nanoTime();
		float[][] lfrFeat = frontend.extract(audioData);
		double tFrontend = (System.nanoTime() - t0) / 1e9;

		// 2. 计算分段（按 LFR 帧切分）
		int chunkFrames = (int) (chunkSize * 100.0 / 6);
		int overlapFrames = (int) (overlap * 100.0 / 6);
		int stride = Math.max(1, chunkFrames - overlapFrames);

		List<TranscriptionResult> allResults = new ArrayList<>();
		for (int start = 0; start < lfrFeat.length; start += stride) {
			int end = Math.min(start + chunkFrames, lfrFeat.length);
			float[][] chunkLfr = Arrays.copyOfRange(lfrFeat, start, end);

			double offsetSec = start * 6 * 0.01;
			TranscriptionResult res = recognizeLfr(chunkLfr, lid, itn, offsetSec, config.getTopK());
			allResults.add(res);

			if (end == lfrFeat.length) break;
		}

		// 3. 结果拼接
		TranscriptionResult merged = mergeResults(allResults, overlap);
		double tTotal = (System.nanoTime() - tStart) / 1e9;

		// 更新总耗时
		return new TranscriptionResult(
			merged.text(),
			merged.results(),
			merged.hotWords(),
			new Timings(tFrontend, merged.timings().encoder(), merged.timings().decoder(),
				merged.timings().radar(), merged.timings().integrate(), tTotal)
		);
	}

	/**
	 * 从 WAV 文件加载音频并识别。
	 *
	 * @param wavPath WAV 文件路径（16kHz 单声道）
	 * @return 转录结果
	 */
	public TranscriptionResult recognizeFile(String wavPath) {
		float[] audio = loadWav(wavPath);
		return recognize(audio);
	}

	/**
	 * 从 WAV 文件加载音频并识别。
	 *
	 * @param wavPath WAV 文件路径
	 * @param lid     语言 ID
	 * @param itn     是否启用 ITN
	 * @return 转录结果
	 */
	public TranscriptionResult recognizeFile(String wavPath, String lid, boolean itn) {
		float[] audio = loadWav(wavPath);
		return recognize(audio, lid, itn, 40, 5);
	}

	// ==================== 底层识别逻辑 ====================

	private TranscriptionResult recognizeLfr(float[][] lfrFeat, String lid, boolean itn,
											 double offsetSec, int topK) {
		try {
			long t0;

			// 1. 编码器推理
			t0 = System.nanoTime();
			float[][][] encOut = encoder.forward(lfrFeat, lid, itn);
			double tEncoder = (System.nanoTime() - t0) / 1e9;

			// 2. 解码器推理
			t0 = System.nanoTime();
			int tValid = lfrFeat.length;
			SenseVoiceDecoder.DecodeResult decodeResult = decoder.decodeAll(
				encOut, tokenizer, topK, tValid);
			double tDecoder = (System.nanoTime() - t0) / 1e9;

			// 3. 热词扫描
			t0 = System.nanoTime();
			List<HotwordRadar.HotwordHit> detectedHotwords = radar.scan(
				decodeResult.radarIndices(), decodeResult.radarProbs(), topK);
			double tRadar = (System.nanoTime() - t0) / 1e9;

			// 4. 整合结果
			t0 = System.nanoTime();
			List<RecognitionResult> integratedList = ResultIntegrator.integrate(
				decodeResult.greedyResults(), detectedHotwords);
			double tIntegrate = (System.nanoTime() - t0) / 1e9;

			// 添加时间偏移
			List<RecognitionResult> recognitionResults = new ArrayList<>();
			for (RecognitionResult item : integratedList) {
				recognitionResults.add(new RecognitionResult(
					item.text(),
					Math.round((item.start() + offsetSec) * 1000.0) / 1000.0,
					item.hotWord()
				));
			}

			StringBuilder textBuilder = new StringBuilder();
			for (RecognitionResult r : recognitionResults) {
				textBuilder.append(r.text());
			}

			List<String> hotwordTexts = new ArrayList<>();
			for (HotwordRadar.HotwordHit h : detectedHotwords) {
				hotwordTexts.add(h.text());
			}

			double tTotal = tEncoder + tDecoder + tRadar + tIntegrate;
			return new TranscriptionResult(
				textBuilder.toString(),
				recognitionResults,
				hotwordTexts,
				new Timings(0, tEncoder, tDecoder, tRadar, tIntegrate, tTotal)
			);
		} catch (OrtException e) {
			throw new RuntimeException("推理失败: " + e.getMessage(), e);
		}
	}

	// ==================== 结果拼接 ====================

	private TranscriptionResult mergeResults(List<TranscriptionResult> resultsList, int overlapSec) {
		if (resultsList.isEmpty()) return null;
		if (resultsList.size() == 1) return resultsList.get(0);

		List<RecognitionResult> mergedResults = new ArrayList<>(resultsList.get(0).results());

		for (int i = 1; i < resultsList.size(); i++) {
			List<RecognitionResult> newRes = resultsList.get(i).results();
			if (newRes.isEmpty()) continue;
			if (mergedResults.isEmpty()) {
				mergedResults.addAll(newRes);
				continue;
			}

			double overlapWindow = overlapSec * 2.0;
			double lastTime = mergedResults.get(mergedResults.size() - 1).start();

			// 提取重叠区域
			List<Integer> prevOverlapIndices = new ArrayList<>();
			for (int idx = 0; idx < mergedResults.size(); idx++) {
				if (mergedResults.get(idx).start() >= lastTime - overlapWindow) {
					prevOverlapIndices.add(idx);
				}
			}
			List<Integer> newOverlapIndices = new ArrayList<>();
			double newStart = newRes.get(0).start();
			for (int idx = 0; idx < newRes.size(); idx++) {
				if (newRes.get(idx).start() <= newStart + overlapWindow) {
					newOverlapIndices.add(idx);
				}
			}

			StringBuilder prevOverlapText = new StringBuilder();
			for (int idx : prevOverlapIndices) {
				prevOverlapText.append(mergedResults.get(idx).text());
			}
			StringBuilder newOverlapText = new StringBuilder();
			for (int idx : newOverlapIndices) {
				newOverlapText.append(newRes.get(idx).text());
			}

			// 寻找最长公共子序列（简化版：直接基于时间戳拼接）
			double lastT = mergedResults.get(mergedResults.size() - 1).start();
			int newStartIdx = 0;
			for (int idx = 0; idx < newRes.size(); idx++) {
				if (newRes.get(idx).start() > lastT) {
					newStartIdx = idx;
					break;
				}
				newStartIdx = newRes.size();
			}
			mergedResults.addAll(newRes.subList(newStartIdx, newRes.size()));
		}

		// 空格切分
		List<RecognitionResult> expandedResults = new ArrayList<>();
		for (RecognitionResult r : mergedResults) {
			String[] parts = r.text().split(" ", -1);
			for (int p = 0; p < parts.length; p++) {
				if (p > 0) {
					expandedResults.add(new RecognitionResult(" ", r.start(), false));
				}
				if (!parts[p].isEmpty()) {
					expandedResults.add(new RecognitionResult(parts[p], r.start(), r.hotWord()));
				}
			}
		}

		// 汇聚所有热词并去重
		List<String> allHotwords = new ArrayList<>();
		for (TranscriptionResult r : resultsList) {
			allHotwords.addAll(r.hotWords());
		}
		List<String> uniqueHotwords = new ArrayList<>(new LinkedHashSet<>(allHotwords));

		StringBuilder text = new StringBuilder();
		for (RecognitionResult r : expandedResults) {
			text.append(r.text());
		}

		return new TranscriptionResult(text.toString(), expandedResults, uniqueHotwords, Timings.empty());
	}

	// ==================== WAV 加载 ====================

	/**
	 * 加载 WAV 文件为 float32 PCM 数据（自动重采样到 16kHz 单声道）。
	 */
	public static float[] loadWav(String wavPath) {
		try {
			File file = new File(wavPath);
			if (!file.exists()) {
				throw new FileNotFoundException("音频文件不存在: " + wavPath);
			}

			AudioInputStream ais = AudioSystem.getAudioInputStream(file);
			AudioFormat format = ais.getFormat();
			int sampleRate = (int) format.getSampleRate();
			int channels = format.getChannels();
			int sampleSizeInBits = format.getSampleSizeInBits();

			// 读取所有字节
			byte[] allBytes = ais.readAllBytes();
			ais.close();

			// 转换为 float32
			float[] audio;
			if (sampleSizeInBits == 16) {
				ByteBuffer buf = ByteBuffer.wrap(allBytes).order(ByteOrder.LITTLE_ENDIAN);
				int numSamples = allBytes.length / 2;
				short[] samples = new short[numSamples];
				for (int i = 0; i < numSamples; i++) {
					samples[i] = buf.getShort();
				}
				// 转单声道
				if (channels > 1) {
					int monoLen = numSamples / channels;
					float[] mono = new float[monoLen];
					for (int i = 0; i < monoLen; i++) {
						float sum = 0;
						for (int c = 0; c < channels; c++) {
							sum += samples[i * channels + c] / 32768.0f;
						}
						mono[i] = sum / channels;
					}
					audio = mono;
				} else {
					audio = new float[numSamples];
					for (int i = 0; i < numSamples; i++) {
						audio[i] = samples[i] / 32768.0f;
					}
				}
			} else if (sampleSizeInBits == 32 && format.getEncoding() == AudioFormat.Encoding.PCM_FLOAT) {
				ByteBuffer buf = ByteBuffer.wrap(allBytes).order(ByteOrder.LITTLE_ENDIAN);
				int numSamples = allBytes.length / 4;
				audio = new float[numSamples];
				for (int i = 0; i < numSamples; i++) {
					audio[i] = buf.getFloat();
				}
				if (channels > 1) {
					int monoLen = numSamples / channels;
					float[] mono = new float[monoLen];
					for (int i = 0; i < monoLen; i++) {
						float sum = 0;
						for (int c = 0; c < channels; c++) {
							sum += audio[i * channels + c];
						}
						mono[i] = sum / channels;
					}
					audio = mono;
				}
			} else {
				throw new UnsupportedOperationException(
					"不支持的音频格式: " + sampleSizeInBits + "bit " + format.getEncoding());
			}

			// 重采样到 16kHz
			if (sampleRate != 16000) {
				audio = resamplePoly(audio, 16000, sampleRate);
			}

			return audio;
		} catch (Exception e) {
			throw new RuntimeException("加载 WAV 文件失败: " + e.getMessage(), e);
		}
	}

	/**
	 * 简易线性重采样（resample_poly 近似）。
	 */
	private static float[] resamplePoly(float[] x, int up, int down) {
		int g = gcd(up, down);
		up /= g;
		down /= g;
		if (up == down) return x.clone();

		int lengthIn = x.length;
		int lengthOut = (int) Math.ceil((double) lengthIn * up / down);

		// FIR 滤波器设计
		int maxRate = Math.max(up, down);
		double fc = 1.0 / maxRate;
		int halfLen = 10 * maxRate;
		int nTaps = 2 * halfLen + 1;

		float[] h = new float[nTaps];
		double beta = 5.0;
		double i0Beta = besselI0(beta);
		for (int i = 0; i < nTaps; i++) {
			double t = i - halfLen;
			double sinc = (t == 0) ? 1.0 : Math.sin(Math.PI * fc * t) / (Math.PI * fc * t);
			double arg = beta * Math.sqrt(1.0 - Math.pow(2.0 * t / (nTaps - 1), 2));
			h[i] = (float) (sinc * besselI0(arg) / i0Beta * up);
		}

		// 上采样 + 滤波 + 下采样
		float[] xUp = new float[lengthIn * up + nTaps];
		for (int i = 0; i < lengthIn; i++) {
			xUp[i * up] = x[i];
		}

		float[] yFull = convolve(xUp, h);
		int offset = (nTaps - 1) / 2;
		float[] y = new float[lengthOut];
		for (int i = 0; i < lengthOut; i++) {
			int idx = offset + i * down;
			y[i] = (idx < yFull.length) ? yFull[idx] : 0;
		}
		return y;
	}

	private static float[] convolve(float[] a, float[] b) {
		int len = a.length + b.length - 1;
		float[] result = new float[len];
		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < b.length; j++) {
				result[i + j] += a[i] * b[j];
			}
		}
		return result;
	}

	private static double besselI0(double x) {
		double sum = 1.0;
		double term = 1.0;
		double xHalf = x / 2.0;
		for (int k = 1; k <= 20; k++) {
			term *= (xHalf / k) * (xHalf / k);
			sum += term;
			if (term < 1e-12 * sum) break;
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

	// ==================== 生命周期 ====================

	@Override
	public void close() {
		if (!closed) {
			try {
				encoder.close();
			} catch (Exception e) {
				log.debug("关闭 encoder 失败: {}", e.getMessage());
			}
			try {
				decoder.close();
			} catch (Exception e) {
				log.debug("关闭 decoder 失败: {}", e.getMessage());
			}
			closed = true;
			log.info("SenseVoice 引擎已关闭");
		}
	}

	private void requireOpen() {
		if (closed) {
			throw new IllegalStateException("SenseVoice has been closed and can no longer be used.");
		}
	}

	private static void requireFile(String path, String name) {
		if (path == null) {
			throw new IllegalArgumentException(name + " is null");
		}
		if (!Files.isRegularFile(Path.of(path))) {
			throw new IllegalArgumentException(name + ": file not found: " + path);
		}
	}
}
