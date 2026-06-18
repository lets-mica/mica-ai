/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent.config;

import lombok.Getter;

import java.nio.file.Path;
import java.util.Collections;
import java.util.List;

/**
 * BERT 意图识别配置。
 *
 * <p>使用 Builder 模式构建，支持链式调用。
 *
 * <p>示例：
 * <pre>{@code
 * BertIntentConfig config = BertIntentConfig.builder()
 *     .modelPath("bert_intent.onnx")
 *     .vocabPath("vocab.txt")
 *     .labels(List.of("weather", "music", "chat"))
 *     .build();
 * }</pre>
 */
@Getter
public final class BertIntentConfig {

	/** 默认最大序列长度 */
	public static final int DEFAULT_MAX_LENGTH = 128;

	/** 默认线程数 */
	public static final int DEFAULT_NUM_THREADS = 1;

	/** ONNX 模型文件路径（必填） */
	private final String modelPath;

	/** 词表文件路径（必填） */
	private final String vocabPath;

	/** 最大序列长度 */
	private final int maxLength;

	/** 意图标签列表，按 ONNX 输出 logits 维度顺序排列 */
	private final List<String> labels;

	/** ONNX 内部线程数 */
	private final int intraOpNumThreads;

	/** ONNX 交互线程数 */
	private final int interOpNumThreads;

	private BertIntentConfig(Builder builder) {
		this.modelPath = builder.modelPath;
		this.vocabPath = builder.vocabPath;
		this.maxLength = builder.maxLength;
		this.labels = builder.labels != null
			? Collections.unmodifiableList(builder.labels)
			: Collections.emptyList();
		this.intraOpNumThreads = builder.intraOpNumThreads;
		this.interOpNumThreads = builder.interOpNumThreads;
	}

	public static Builder builder() {
		return new Builder();
	}

	// --- Getters ---

	/**
	 * Builder。
	 */
	public static final class Builder {
		private String modelPath;
		private String vocabPath;
		private int maxLength = DEFAULT_MAX_LENGTH;
		private List<String> labels;
		private int intraOpNumThreads = DEFAULT_NUM_THREADS;
		private int interOpNumThreads = DEFAULT_NUM_THREADS;

		public Builder modelPath(String modelPath) {
			this.modelPath = modelPath;
			return this;
		}

		public Builder modelPath(Path modelPath) {
			this.modelPath = modelPath.toString();
			return this;
		}

		public Builder vocabPath(String vocabPath) {
			this.vocabPath = vocabPath;
			return this;
		}

		public Builder vocabPath(Path vocabPath) {
			this.vocabPath = vocabPath.toString();
			return this;
		}

		public Builder maxLength(int maxLength) {
			this.maxLength = maxLength;
			return this;
		}

		public Builder labels(List<String> labels) {
			this.labels = labels;
			return this;
		}

		public Builder intraOpNumThreads(int intraOpNumThreads) {
			this.intraOpNumThreads = intraOpNumThreads;
			return this;
		}

		public Builder interOpNumThreads(int interOpNumThreads) {
			this.interOpNumThreads = interOpNumThreads;
			return this;
		}

		public BertIntentConfig build() {
			if (modelPath == null || modelPath.isBlank()) {
				throw new IllegalArgumentException("modelPath is required");
			}
			if (vocabPath == null || vocabPath.isBlank()) {
				throw new IllegalArgumentException("vocabPath is required");
			}
			if (maxLength < 3) {
				throw new IllegalArgumentException("maxLength must be at least 3 (CLS + 1 token + SEP)");
			}
			return new BertIntentConfig(this);
		}
	}
}
