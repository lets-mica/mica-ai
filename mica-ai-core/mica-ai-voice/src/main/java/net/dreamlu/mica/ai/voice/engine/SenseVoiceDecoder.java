package net.dreamlu.mica.ai.voice.engine;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;

import ai.onnxruntime.OnnxJavaType;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.*;

/**
 * SenseVoice ONNX CTC 解码器。
 *
 * <p>对应 Python 端 {@code SenseVoiceDecoder}：
 * <ul>
 *   <li>接收编码器输出 enc_out (1, T+4, 512)</li>
 *   <li>执行 CTC Head 推理，输出 Top-K 概率和索引</li>
 *   <li>Greedy 解码 + CTC collapse</li>
 * </ul>
 */
@Slf4j
public final class SenseVoiceDecoder implements AutoCloseable {

	private static final int PROMPT_LEN = 4;
	private static final int BLANK_ID = 0;
	/** 每帧对应的时间步长（秒）: 1帧 = 0.06s */
	private static final double FRAME_DURATION = 0.060;

	private final OrtEnvironment env;
	private final OrtSession session;

	public SenseVoiceDecoder(String decoderPath, OrtEnvironment env) throws OrtException {
		this.env = env;

		OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
		opts.setIntraOpNumThreads(1);
		opts.setInterOpNumThreads(1);
		opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

		this.session = env.createSession(decoderPath, opts);
		log.info("[Decoder] 初始化完成");
	}

	/**
	 * 执行解码并返回 Greedy 结果。
	 *
	 * @param encOut    编码器输出 [1][T+4][512]
	 * @param tokenizer SentencePiece 分词器
	 * @param topK      Top-K 深度（用于热词雷达）
	 * @param tValid    有效帧数（不含填充）
	 * @return 解码结果
	 */
	public DecodeResult decodeAll(float[][][] encOut, SentencePieceTokenizer tokenizer,
								  int topK, int tValid) throws OrtException {
		int tPlus4 = encOut[0].length;
		int hiddenSize = encOut[0][0].length;

		// 构造输入张量
		long[] shape = {1, tPlus4, hiddenSize};
		FloatBuffer buf = FloatBuffer.allocate(tPlus4 * hiddenSize);
		for (float[] row : encOut[0]) {
			buf.put(row);
		}
		buf.flip();

		Map<String, OnnxTensor> inputs = Collections.singletonMap(
			"enc_out", OnnxTensor.createTensor(env, buf, shape));

		float[][] topkLogProbs;
		int[][] topkIndices;

		try (OrtSession.Result result = session.run(inputs)) {
			OnnxTensor logProbTensor = (OnnxTensor) result.get(0);
			OnnxTensor indexTensor = (OnnxTensor) result.get(1);
			topkLogProbs = readFloat2D(logProbTensor);
			topkIndices = readInt2D(indexTensor);
		}

		// topkLogProbs / topkIndices 形状均为 [T+4][K]（batch 维度已展平）
		// 确定有效范围（跳过 Prompt 区域）
		int start = PROMPT_LEN;
		int end = Math.min(tValid + PROMPT_LEN, topkIndices.length);
		int validLen = end - start;

		// 提取 Top-K 空间用于雷达
		int totalK = topkIndices[0].length;
		int actualK = Math.min(topK, totalK);
		int[][] radarIndices = new int[validLen][actualK];
		float[][] radarProbs = new float[validLen][actualK];
		int[] top1Indices = new int[validLen];

		for (int t = 0; t < validLen; t++) {
			for (int k = 0; k < actualK; k++) {
				radarIndices[t][k] = topkIndices[start + t][k];
				radarProbs[t][k] = (float) Math.exp(topkLogProbs[start + t][k]);
			}
			top1Indices[t] = radarIndices[t][0];
		}

		// CTC Greedy 解码（collapse 重复 + 过滤 blank）
		List<GreedyItem> greedyResults = new ArrayList<>();
		if (validLen > 0) {
			int currId = top1Indices[0];
			int startFrame = 0;

			for (int i = 1; i < validLen; i++) {
				if (top1Indices[i] != currId) {
					if (currId != BLANK_ID) {
						String piece = tokenizer.idToPiece(currId);
						String text = piece.replace('\u2581', ' ');
						if (!text.isBlank() || " ".equals(text)) {
							greedyResults.add(new GreedyItem(text, Math.round(startFrame * FRAME_DURATION * 1000.0) / 1000.0));
						}
					}
					currId = top1Indices[i];
					startFrame = i;
				}
			}
			// 最后一个
			if (currId != BLANK_ID) {
				String piece = tokenizer.idToPiece(currId);
				String text = piece.replace('\u2581', ' ');
				if (!text.isBlank() || " ".equals(text)) {
					greedyResults.add(new GreedyItem(text, Math.round(startFrame * FRAME_DURATION * 1000.0) / 1000.0));
				}
			}
		}

		return new DecodeResult(greedyResults, radarIndices, radarProbs, top1Indices);
	}

	@Override
	public void close() throws OrtException {
		if (session != null) {
			session.close();
		}
	}

	/**
	 * Greedy 解码条目。
	 */
	public record GreedyItem(String text, double start) {
	}

	/**
	 * 解码结果。
	 */
	public record DecodeResult(List<GreedyItem> greedyResults,
							   int[][] radarIndices,
							   float[][] radarProbs,
							   int[] top1Indices) {
	}

	// ==================== Tensor 读取 ====================

	private static float[][] readFloat2D(OnnxTensor tensor) throws OrtException {
		FloatBuffer buf = tensor.getFloatBuffer();
		long[] shape = tensor.getInfo().getShape();
		if (shape.length == 3) {
			// (1, T, K) → 取 batch=0
			int t = (int) shape[1];
			int k = (int) shape[2];
			float[] data = new float[t * k];
			buf.get(data);
			float[][] result = new float[t][k];
			for (int i = 0; i < t; i++) {
				System.arraycopy(data, i * k, result[i], 0, k);
			}
			return result;
		}
		int rows = (int) shape[0];
		int cols = (int) shape[1];
		float[] data = new float[rows * cols];
		buf.get(data);
		float[][] result = new float[rows][cols];
		for (int i = 0; i < rows; i++) {
			System.arraycopy(data, i * cols, result[i], 0, cols);
		}
		return result;
	}

	private static int[][] readInt2D(OnnxTensor tensor) throws OrtException {
		long[] shape = tensor.getInfo().getShape();
		int totalElements = 1;
		for (long s : shape) totalElements *= (int) s;

		long[] longData;
		OnnxJavaType type = tensor.getInfo().type;
		if (type == OnnxJavaType.INT64) {
			LongBuffer longBuf = tensor.getLongBuffer();
			longData = new long[totalElements];
			longBuf.get(longData);
		} else if (type == OnnxJavaType.INT32) {
			IntBuffer intBuf = tensor.getIntBuffer();
			longData = new long[totalElements];
			for (int i = 0; i < totalElements; i++) {
				longData[i] = intBuf.get();
			}
		} else {
			// fallback: try to read as flat value array
			var value = tensor.getValue();
			if (value instanceof long[][][]) {
				long[][][] arr = (long[][][]) value;
				int t = arr[0].length;
				int k = arr[0][0].length;
				int[][] result = new int[t][k];
				for (int i = 0; i < t; i++) {
					for (int j = 0; j < k; j++) {
						result[i][j] = (int) arr[0][i][j];
					}
				}
				return result;
			}
			throw new RuntimeException("Unsupported tensor type for int read: " + type);
		}

		if (shape.length == 3) {
			int t = (int) shape[1];
			int k = (int) shape[2];
			int[][] result = new int[t][k];
			for (int i = 0; i < t; i++) {
				for (int j = 0; j < k; j++) {
					result[i][j] = (int) longData[i * k + j];
				}
			}
			return result;
		}

		int rows = (int) shape[0];
		int cols = (int) shape[1];
		int[][] result = new int[rows][cols];
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				result[i][j] = (int) longData[i * cols + j];
			}
		}
		return result;
	}
}

