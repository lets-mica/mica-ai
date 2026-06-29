/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent.engine;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.intent.config.BertIntentConfig;
import net.dreamlu.mica.ai.intent.config.IntentResult;
import net.dreamlu.mica.ai.intent.tokenizer.BertTokenizer;
import net.dreamlu.mica.ai.intent.tokenizer.VocabLoader;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * BERT 中文意图识别入口。
 *
 * <p>组合 {@link BertTokenizer} 和 {@link BertIntentEngine}，提供简洁的预测 API。
 *
 * <p>典型用法：
 * <pre>{@code
 * BertIntentConfig config = BertIntentConfig.builder()
 *     .modelPath("bert_intent.onnx")
 *     .vocabPath("vocab.txt")
 *     .labels(List.of("weather", "music", "chat", "news"))
 *     .build();
 *
 * BertIntent intent = new BertIntent(config);
 *
 * IntentResult result = intent.predict("今天天气怎么样");
 * System.out.println(result.getIntent() + " -> " + result.getConfidence());
 *
 * intent.close();
 * }</pre>
 */
@Slf4j
public class BertIntent implements Closeable {

	private final BertTokenizer tokenizer;
	private final BertIntentEngine engine;
	private boolean closed = false;

	/**
	 * 从配置构建意图识别器。
	 *
	 * @param config 意图识别配置
	 * @throws IOException 词表文件读取失败
	 */
	public BertIntent(BertIntentConfig config) throws IOException {
		VocabLoader vocab = VocabLoader.load(config.getVocabPath());
		this.tokenizer = new BertTokenizer(vocab, config.getMaxLength());
		this.engine = new BertIntentEngine(config);
		log.info("BertIntent 初始化完成, maxLength={}, labels={}",
			config.getMaxLength(), config.getLabels());
	}

	/**
	 * 预测文本的意图。
	 *
	 * @param text 输入中文文本
	 * @return 意图 + 置信度
	 */
	public IntentResult predict(String text) {
		requireOpen();

		// 1. 分词
		BertTokenizer.TokenResult tokens = tokenizer.tokenize(text);

		// 2. 推理
		return engine.inference(tokens.inputIds(), tokens.attentionMask(), tokens.tokenTypeIds());
	}

	/**
	 * 批量预测（顺序执行，非批量推理）。
	 *
	 * @param texts 输入文本列表
	 * @return 结果列表，与输入顺序一致
	 */
	public List<IntentResult> predictBatch(List<String> texts) {
		requireOpen();
		List<IntentResult> results = new ArrayList<>(texts.size());
		for (String text : texts) {
			results.add(predict(text));
		}
		return results;
	}

	// ==================== 生命周期 ====================

	@Override
	public void close() {
		if (!closed) {
			engine.close();
			closed = true;
			log.info("BertIntent 已关闭");
		}
	}

	private void requireOpen() {
		if (closed) {
			throw new IllegalStateException("BertIntent 已关闭，无法继续使用");
		}
	}
}
