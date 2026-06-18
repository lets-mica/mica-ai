/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent;

import net.dreamlu.mica.ai.intent.config.BertIntentConfig;
import net.dreamlu.mica.ai.intent.config.IntentResult;
import net.dreamlu.mica.ai.intent.engine.BertIntent;
import org.junit.jupiter.api.*;
import org.junit.jupiter.api.Assumptions;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link BertIntent} 端到端集成测试。
 *
 * <p>需要实际 ONNX 模型文件，缺失时自动跳过（SKIPPED）。
 *
 * <p>模型准备：
 * <ol>
 *   <li>下载 hfl/chinese-bert-wwm-ext 模型</li>
 *   <li>添加分类头后导出 ONNX：
 *     <pre>
 * python export_bert_intent.py \
 *   --model_name hfl/chinese-bert-wwm-ext \
 *   --num_labels 4 \
 *   --output bert_intent.onnx \
 *   --max_length 128
 *     </pre>
 *   </li>
 *   <li>将 bert_intent.onnx 和 vocab.txt 放入指定路径</li>
 * </ol>
 */
@DisplayName("BertIntent 集成测试（需 ONNX 模型）")
class BertIntentIntegrationTest {

	/**
	 * 模型文件路径 —— 修改为实际的模型路径。
	 * 常见位置：
	 * <ul>
	 *   <li>E:\\codes\\ai\\Intent-ONNX\\model\\bert_intent.onnx</li>
	 *   <li>E:\\codes\\ai\\chinese-bert-wwm-ext\\bert_intent.onnx</li>
	 * </ul>
	 */
	private static final String[] MODEL_PATH_CANDIDATES = {
		"E:\\codes\\ai\\ai_test\\Intent-ONNX\\model\\bert_intent.onnx",
		"E:\\codes\\ai\\Intent-ONNX\\model\\bert_intent.onnx",
		"E:\\codes\\ai\\chinese-bert-wwm-ext\\bert_intent.onnx",
		"./bert_intent.onnx",
	};

	/**
	 * 词表文件路径候选。
	 */
	private static final String[] VOCAB_PATH_CANDIDATES = {
		"E:\\codes\\ai\\ai_test\\Intent-ONNX\\model\\vocab.txt",
		"E:\\codes\\ai\\Intent-ONNX\\model\\vocab.txt",
		"E:\\codes\\ai\\chinese-bert-wwm-ext\\vocab.txt",
		"./vocab.txt",
	};

	/**
	 * 默认意图标签（按 ONNX 输出顺序）。
	 */
	private static final List<String> DEFAULT_LABELS = List.of(
		"weather", "music", "chat", "news"
	);

	private BertIntent bertIntent;
	private String resolvedModelPath;
	private String resolvedVocabPath;

	@BeforeEach
	void setUp() throws IOException {
		resolvedModelPath = findFirst(MODEL_PATH_CANDIDATES);
		resolvedVocabPath = findFirst(VOCAB_PATH_CANDIDATES);

		Assumptions.assumeTrue(resolvedModelPath != null,
			"跳过：未找到 bert_intent.onnx 模型文件。请将模型放到 E:\\codes\\ai\\Intent-ONNX\\model\\ 下");
		Assumptions.assumeTrue(resolvedVocabPath != null,
			"跳过：未找到 vocab.txt 词表文件");

		BertIntentConfig config = BertIntentConfig.builder()
			.modelPath(resolvedModelPath)
			.vocabPath(resolvedVocabPath)
			.maxLength(128)
			.labels(DEFAULT_LABELS)
			.build();

		bertIntent = new BertIntent(config);
	}

	@AfterEach
	void tearDown() {
		if (bertIntent != null) {
			bertIntent.close();
		}
	}

	// ==================== 推理测试 ====================

	@Test
	@DisplayName("单条中文意图预测 → 返回非空结果")
	void testPredictChineseText() {
		IntentResult result = bertIntent.predict("今天天气怎么样");

		assertNotNull(result);
		assertNotNull(result.intent());
		assertTrue(result.confidence() >= 0.0f && result.confidence() <= 1.0f,
			"置信度应在 [0, 1] 范围内");
		assertNotNull(result.allScores());
		assertEquals(DEFAULT_LABELS.size(), result.allScores().size(),
			"allScores 大小应等于标签数");
	}

	@Test
	@DisplayName("所有标签都出现在 allScores 中")
	void testAllScoresContainsAllLabels() {
		IntentResult result = bertIntent.predict("播放一首歌");

		for (String label : DEFAULT_LABELS) {
			assertTrue(result.allScores().containsKey(label),
				"allScores 应包含标签: " + label);
		}
	}

	@Test
	@DisplayName("概率和为 1（softmax 验证）")
	void testProbabilitySumToOne() {
		IntentResult result = bertIntent.predict("今天天气怎么样");

		float sum = 0f;
		for (float score : result.allScores().values()) {
			sum += score;
		}
		assertEquals(1.0f, sum, 0.01f, "softmax 概率和应为 1");
	}

	@Test
	@DisplayName("confidence 等于 max(allScores)")
	void testConfidenceEqualsMaxScore() {
		IntentResult result = bertIntent.predict("帮我查一下新闻");

		float maxScore = result.allScores().values().stream()
			.max(Float::compare)
			.orElse(0f);

		assertEquals(maxScore, result.confidence(), 0.0001f,
			"confidence 应等于最大概率值");
	}

	// ==================== 批量推理 ====================

	@Test
	@DisplayName("批量预测返回等长结果")
	void testPredictBatch() {
		List<String> texts = List.of(
			"今天天气怎么样",
			"播放一首音乐",
			"你好",
			"有什么新闻"
		);

		List<IntentResult> results = bertIntent.predictBatch(texts);

		assertEquals(texts.size(), results.size());
		for (IntentResult result : results) {
			assertNotNull(result.intent());
			assertTrue(result.confidence() >= 0.0f && result.confidence() <= 1.0f);
		}
	}

	// ==================== 生命周期 ====================

	@Test
	@DisplayName("关闭后调用 predict 抛 IllegalStateException")
	void testCloseThenPredict() {
		bertIntent.close();

		assertThrows(IllegalStateException.class, () ->
			bertIntent.predict("你好")
		);
	}

	@Test
	@DisplayName("重复 close 不抛异常")
	void testDoubleClose() {
		bertIntent.close();
		assertDoesNotThrow(() -> bertIntent.close());
	}

	// ==================== 助手方法 ====================

	/**
	 * 从候选路径中查找第一个存在的文件。
	 */
	private static String findFirst(String[] candidates) {
		for (String path : candidates) {
			if (path != null && Files.isRegularFile(Path.of(path))) {
				System.out.println("[INTEGRATION-TEST] 找到模型: " + path);
				return path;
			}
		}
		return null;
	}
}
