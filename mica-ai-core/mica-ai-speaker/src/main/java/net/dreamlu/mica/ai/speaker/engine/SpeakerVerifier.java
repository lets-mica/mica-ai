/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.speaker.engine;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.utils.AudioUtils;

import java.io.Closeable;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.*;

/**
 * ERes2Net-base 声纹识别引擎（纯 ONNX Runtime + JTransforms）。
 *
 * <p>使用 speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx 模型进行声纹验证。
 *
 * <p>模型规格：
 * <ul>
 *   <li>输入：feats → float32 [1, T, 80]（log-Mel FBank）</li>
 *   <li>输出：embedding → float32 [1, 192]</li>
 *   <li>推理后自动 L2 归一化</li>
 * </ul>
 *
 * <p>典型用法：
 * <pre>{@code
 * SpeakerVerifier verifier = new SpeakerVerifier(Path.of("model.onnx"));
 *
 * float[] emb1 = verifier.extractEmbedding(Path.of("speaker1.wav"));
 * float[] emb2 = verifier.extractEmbedding(Path.of("speaker2.wav"));
 *
 * float score = SpeakerVerifier.cosineSimilarity(emb1, emb2);
 * boolean samePerson = score > 0.58;
 *
 * // 声纹注册（多段音频取平均）
 * float[] enrolled = verifier.enrollSpeaker(List.of(
 *     Path.of("s1_enroll_1.wav"),
 *     Path.of("s1_enroll_2.wav"),
 *     Path.of("s1_enroll_3.wav")
 * ));
 *
 * verifier.close();
 * }</pre>
 */
@Slf4j
public class SpeakerVerifier implements Closeable, AutoCloseable {

	/**
	 * 默认余弦相似度阈值（可根据实际场景调整）。
	 */
	public static final float DEFAULT_THRESHOLD = 0.58f;

	/**
	 * Embedding 维度。
	 */
	public static final int EMBEDDING_DIM = 192;

	/**
	 * Mel 频带数。
	 */
	public static final int MEL_BINS = 80;

	private final OrtEnvironment env;
	private final OrtSession session;
	private final FBankExtractor frontend;
	private final String inputName;
	private final String outputName;
	private boolean closed = false;

	/**
	 * 从文件路径加载 ONNX 模型。
	 *
	 * @param modelPath ONNX 模型文件路径
	 */
	public SpeakerVerifier(Path modelPath) {
		try {
			this.env = OrtEnvironment.getEnvironment();
			OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
			opts.setIntraOpNumThreads(1);
			opts.setInterOpNumThreads(1);
			opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
			this.session = env.createSession(modelPath.toString(), opts);

			// 读取输入/输出名称
			this.inputName = session.getInputInfo().keySet().iterator().next();
			this.outputName = session.getOutputInfo().keySet().iterator().next();

			this.frontend = new FBankExtractor();

			log.info("SpeakerVerifier 初始化完成, 输入: {} [1,T,{}], 输出: {} [1,{}]",
				inputName, MEL_BINS, outputName, EMBEDDING_DIM);
		} catch (OrtException e) {
			throw new RuntimeException("加载 ONNX 模型失败: " + modelPath, e);
		}
	}

	/**
	 * 便捷构造：接受字符串路径。
	 */
	public SpeakerVerifier(String modelPath) {
		this(Path.of(modelPath));
	}

	// ==================== 核心 API ====================

	/**
	 * 从 WAV 文件提取声纹 embedding（16kHz/16bit/单声道）。
	 *
	 * @param wavPath WAV 文件路径
	 * @return L2 归一化后的 192 维 embedding
	 * @throws IOException 文件读取失败或格式不支持
	 */
	public float[] extractEmbedding(Path wavPath) throws IOException {
		float[] audio = AudioUtils.loadWavAsFloat(wavPath);
		return extractEmbedding(audio);
	}

	/**
	 * 从 float32 PCM 音频数据提取声纹 embedding。
	 *
	 * @param audio 16kHz 单声道 float32 PCM 数据
	 * @return L2 归一化后的 192 维 embedding
	 */
	public float[] extractEmbedding(float[] audio) {
		requireOpen();

		// 1. 提取 FBank 特征 [T, 80]
		float[][] feats = frontend.extract(audio);
		int T = feats.length;

		// 2. 展平为 [1, T, 80] 并写入 FloatBuffer
		int totalSize = 1 * T * MEL_BINS;
		FloatBuffer featBuf = FloatBuffer.allocate(totalSize);
		for (int t = 0; t < T; t++) {
			featBuf.put(feats[t]);
		}
		featBuf.flip(); // 切换到读模式

		// 3. ONNX 推理
		long[] featShape = {1, T, MEL_BINS};
		try (OnnxTensor tensor = OnnxTensor.createTensor(env, featBuf, featShape);
			 OrtSession.Result result = session.run(
				 Collections.singletonMap(inputName, tensor))) {

			// 4. 提取 embedding [1, 192] → float[192]
			OnnxTensor outTensor = (OnnxTensor) result.get(outputName).get();
			float[] embedding = new float[EMBEDDING_DIM];
			outTensor.getFloatBuffer().get(embedding);

			// 5. L2 归一化
			l2Normalize(embedding);

			return embedding;
		} catch (OrtException e) {
			throw new RuntimeException("ONNX 推理失败: " + e.getMessage(), e);
		}
	}

	/**
	 * 计算两个 embedding 的余弦相似度。
	 *
	 * @param a 第一个 embedding
	 * @param b 第二个 embedding
	 * @return 余弦相似度 [-1, 1]，值越大表示越相似
	 */
	public static float cosineSimilarity(float[] a, float[] b) {
		if (a.length != b.length) {
			throw new IllegalArgumentException(
				"Embedding 维度不一致: " + a.length + " vs " + b.length);
		}

		double dot = 0;
		double normA = 0;
		double normB = 0;
		for (int i = 0; i < a.length; i++) {
			dot += (double) a[i] * b[i];
			normA += (double) a[i] * a[i];
			normB += (double) b[i] * b[i];
		}

		double denominator = Math.sqrt(normA) * Math.sqrt(normB);
		if (denominator < 1e-12) {
			return 0f;
		}
		return (float) (dot / denominator);
	}

	// ==================== 声纹注册 ====================

	/**
	 * 从多段音频注册声纹（多个 embedding 取平均 + L2 归一化）。
	 *
	 * @param wavPaths 同一说话人的多段注册音频
	 * @return 注册后的 192 维声纹 embedding
	 * @throws IOException 文件读取失败
	 */
	public float[] enrollSpeaker(List<Path> wavPaths) throws IOException {
		if (wavPaths == null || wavPaths.isEmpty()) {
			throw new IllegalArgumentException("注册音频列表不能为空");
		}

		List<float[]> embeddings = new ArrayList<>(wavPaths.size());
		for (Path p : wavPaths) {
			embeddings.add(extractEmbedding(p));
		}
		return averageEmbeddings(embeddings);
	}

	/**
	 * 从多段 PCM 数据注册声纹。
	 *
	 * @param audioList 同一说话人的多段 16kHz 单声道 PCM 数据
	 * @return 注册后的 192 维声纹 embedding
	 */
	public float[] enrollSpeakerFromPcm(List<float[]> audioList) {
		if (audioList == null || audioList.isEmpty()) {
			throw new IllegalArgumentException("注册音频列表不能为空");
		}

		List<float[]> embeddings = new ArrayList<>(audioList.size());
		for (float[] audio : audioList) {
			embeddings.add(extractEmbedding(audio));
		}
		return averageEmbeddings(embeddings);
	}

	/**
	 * 验证给定音频是否匹配已注册声纹。
	 *
	 * @param enrolled  已注册的声纹 embedding
	 * @param testWav   待验证的 WAV 文件
	 * @param threshold 判定阈值（默认 0.58）
	 * @return true 若相似度 ≥ threshold
	 * @throws IOException 文件读取失败
	 */
	public boolean verify(float[] enrolled, Path testWav, float threshold) throws IOException {
		float[] testEmb = extractEmbedding(testWav);
		float score = cosineSimilarity(enrolled, testEmb);
		log.debug("声纹匹配分数: {}, 阈值: {}", score, threshold);
		return score >= threshold;
	}

	/**
	 * 使用默认阈值验证（0.58）。
	 */
	public boolean verify(float[] enrolled, Path testWav) throws IOException {
		return verify(enrolled, testWav, DEFAULT_THRESHOLD);
	}

	// ==================== 工具方法 ====================

	/**
	 * 多个 embedding 取平均后 L2 归一化。
	 */
	private static float[] averageEmbeddings(List<float[]> embeddings) {
		int dim = embeddings.get(0).length;
		float[] avg = new float[dim];
		for (float[] emb : embeddings) {
			for (int i = 0; i < dim; i++) {
				avg[i] += emb[i];
			}
		}
		for (int i = 0; i < dim; i++) {
			avg[i] /= embeddings.size();
		}
		l2Normalize(avg);
		return avg;
	}

	/**
	 * L2 归一化（就地操作）。
	 */
	public static void l2Normalize(float[] vec) {
		double sumSq = 0;
		for (float v : vec) {
			sumSq += (double) v * v;
		}
		double norm = Math.sqrt(sumSq);
		if (norm > 1e-12) {
			for (int i = 0; i < vec.length; i++) {
				vec[i] = (float) (vec[i] / norm);
			}
		}
	}

	// ==================== 生命周期 ====================

	@Override
	public void close() {
		if (!closed) {
			try {
				session.close();
			} catch (Exception e) {
				log.debug("关闭 ONNX session 异常: {}", e.getMessage());
			}
			try {
				env.close();
			} catch (Exception e) {
				log.debug("关闭 ONNX environment 异常: {}", e.getMessage());
			}
			closed = true;
			log.info("SpeakerVerifier 已关闭");
		}
	}

	private void requireOpen() {
		if (closed) {
			throw new IllegalStateException("SpeakerVerifier 已关闭，无法继续使用");
		}
	}
}
