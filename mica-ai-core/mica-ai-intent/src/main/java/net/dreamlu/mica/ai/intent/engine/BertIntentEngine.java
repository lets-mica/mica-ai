/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent.engine;

import ai.onnxruntime.*;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.intent.config.BertIntentConfig;
import net.dreamlu.mica.ai.intent.config.IntentResult;

import java.io.Closeable;
import java.nio.LongBuffer;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * BERT 意图识别 ONNX 推理引擎。
 *
 * <p>加载 bert_intent.onnx 模型，执行中文意图分类推理。
 *
 * <p>模型规格：
 * <ul>
 *   <li>输入：input_ids / attention_mask / token_type_ids → int64 [1, maxLength]</li>
 *   <li>输出：logits → float32 [1, numLabels]</li>
 * </ul>
 */
@Slf4j
public final class BertIntentEngine implements Closeable {
	private final OrtEnvironment env;
	private final OrtSession session;
	private final int maxLength;
	private final List<String> labels;
	private final int numLabels;
	private boolean closed = false;

	/**
	 * 从配置构建引擎。
	 *
	 * @param config 意图识别配置
	 * @throws RuntimeException ONNX 模型加载失败
	 */
	public BertIntentEngine(BertIntentConfig config) {
		this.maxLength = config.getMaxLength();
		this.labels = config.getLabels();
		this.numLabels = labels != null ? labels.size() : 0;

		try {
			this.env = OrtEnvironment.getEnvironment();
			OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
			opts.setIntraOpNumThreads(config.getIntraOpNumThreads());
			opts.setInterOpNumThreads(config.getInterOpNumThreads());
			opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);

			this.session = env.createSession(config.getModelPath(), opts);
			log.info("加载 BERT 意图模型完成: {}, maxLength={}, numLabels={}",
				config.getModelPath(), maxLength, numLabels);
		} catch (OrtException e) {
			throw new RuntimeException("加载 BERT 意图模型失败: " + config.getModelPath(), e);
		}
	}

	/**
	 * 执行意图推理。
	 *
	 * @param inputIds      token ID 序列 [maxLength]
	 * @param attentionMask 注意力掩码 [maxLength]
	 * @param tokenTypeIds  token 类型 ID [maxLength]
	 * @return 推理结果（意图标签 + 置信度 + 所有分数）
	 */
	public IntentResult inference(long[] inputIds, long[] attentionMask, long[] tokenTypeIds) {
		requireOpen();

		try {
			// 构造三个输入张量，shape 均为 [1, maxLength]
			LongBuffer inputIdsBuf = LongBuffer.wrap(inputIds);
			LongBuffer attentionMaskBuf = LongBuffer.wrap(attentionMask);
			LongBuffer tokenTypeIdsBuf = LongBuffer.wrap(tokenTypeIds);
			long[] shape = {1, maxLength};

			try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIdsBuf, shape);
				 OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMaskBuf, shape);
				 OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(env, tokenTypeIdsBuf, shape)) {

				Map<String, OnnxTensor> inputs = new LinkedHashMap<>(3);
				inputs.put("input_ids", inputIdsTensor);
				inputs.put("attention_mask", attentionMaskTensor);
				inputs.put("token_type_ids", tokenTypeIdsTensor);

				try (OrtSession.Result result = session.run(inputs)) {
					// 获取 logits 输出 [1, numLabels]
					OnnxTensor logitsTensor = (OnnxTensor) result.get(0);
					float[] logits = new float[numLabels];
					logitsTensor.getFloatBuffer().get(logits);

					// Softmax → 概率分布
					float[] probs = softmax(logits);

					// argmax → 最佳意图 + 置信度
					int bestIdx = 0;
					float bestScore = probs[0];
					for (int i = 1; i < probs.length; i++) {
						if (probs[i] > bestScore) {
							bestScore = probs[i];
							bestIdx = i;
						}
					}

					// 构建标签→分数映射
					Map<String, Float> allScores = new LinkedHashMap<>(numLabels);
					for (int i = 0; i < numLabels; i++) {
						String label = labels != null && i < labels.size()
							? labels.get(i)
							: String.valueOf(i);
						allScores.put(label, probs[i]);
					}

					String bestLabel = labels != null && bestIdx < labels.size()
						? labels.get(bestIdx)
						: String.valueOf(bestIdx);

					log.debug("意图推理结果: {} (confidence={})", bestLabel, bestScore);
					return new IntentResult(bestLabel, bestScore, allScores);
				}
			}
		} catch (OrtException e) {
			throw new RuntimeException("BERT 意图推理失败: " + e.getMessage(), e);
		}
	}

	/**
	 * Softmax 归一化。
	 *
	 * <p>为了数值稳定性，先减去最大值。
	 */
	private static float[] softmax(float[] logits) {
		int n = logits.length;
		float[] probs = new float[n];

		// 找最大值
		float max = logits[0];
		for (int i = 1; i < n; i++) {
			if (logits[i] > max) {
				max = logits[i];
			}
		}

		// exp(x - max) + sum
		double sum = 0;
		for (int i = 0; i < n; i++) {
			double v = Math.exp(logits[i] - max);
			probs[i] = (float) v;
			sum += v;
		}

		// 归一化
		for (int i = 0; i < n; i++) {
			probs[i] = (float) (probs[i] / sum);
		}

		return probs;
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
			log.info("BertIntentEngine 已关闭");
		}
	}

	private void requireOpen() {
		if (closed) {
			throw new IllegalStateException("BertIntentEngine 已关闭，无法继续使用");
		}
	}
}
