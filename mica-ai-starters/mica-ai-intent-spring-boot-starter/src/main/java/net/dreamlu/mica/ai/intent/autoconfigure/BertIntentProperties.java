/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent.autoconfigure;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.List;

/**
 * BERT 意图识别配置属性。
 *
 * <p>对应 {@code mica.ai.intent} 配置前缀。
 *
 * <p>示例配置：
 * <pre>
 * mica:
 *   ai:
 *     intent:
 *       model-path: /path/to/bert_intent.onnx
 *       vocab-path: /path/to/vocab.txt
 *       labels:
 *         - weather
 *         - music
 *         - chat
 * </pre>
 */
@Data
@ConfigurationProperties(prefix = "mica.ai.intent")
public class BertIntentProperties {

	/**
	 * 是否启用该 Starter。默认 {@code true}：启用时必填的 BERT 模型/词表/标签缺失将启动失败；
	 * 设为 {@code false} 时整个 Starter 不注入任何 Bean。
	 */
	private boolean enabled = true;

	/** ONNX 模型文件路径（必填） */
	private String modelPath;

	/** BERT 词表文件路径（必填） */
	private String vocabPath;

	/** 最大序列长度，默认 128 */
	private int maxLength = 128;

	/** 意图标签列表，按 ONNX 输出 logits 维度顺序排列（必填） */
	private List<String> labels;

	/** ONNX 内部线程数 */
	private int intraOpNumThreads = 1;

	/** ONNX 交互线程数 */
	private int interOpNumThreads = 1;
}
