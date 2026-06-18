/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent;

import net.dreamlu.mica.ai.intent.config.BertIntentConfig;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link BertIntentConfig} Builder 测试。
 */
@DisplayName("BertIntentConfig")
class BertIntentConfigTest {

	@Test
	@DisplayName("Builder 完整构建")
	void testFullBuilder() {
		BertIntentConfig config = BertIntentConfig.builder()
			.modelPath("bert_intent.onnx")
			.vocabPath("vocab.txt")
			.maxLength(128)
			.labels(List.of("weather", "music", "chat"))
			.intraOpNumThreads(2)
			.interOpNumThreads(2)
			.build();

		assertEquals("bert_intent.onnx", config.getModelPath());
		assertEquals("vocab.txt", config.getVocabPath());
		assertEquals(128, config.getMaxLength());
		assertEquals(List.of("weather", "music", "chat"), config.getLabels());
		assertEquals(2, config.getIntraOpNumThreads());
		assertEquals(2, config.getInterOpNumThreads());
	}

	@Test
	@DisplayName("默认值正确")
	void testDefaultValues() {
		BertIntentConfig config = BertIntentConfig.builder()
			.modelPath("model.onnx")
			.vocabPath("vocab.txt")
			.build();

		assertEquals(BertIntentConfig.DEFAULT_MAX_LENGTH, config.getMaxLength(),
			"默认 maxLength = 128");
		assertEquals(BertIntentConfig.DEFAULT_NUM_THREADS, config.getIntraOpNumThreads(),
			"默认线程数 = 1");
		assertEquals(BertIntentConfig.DEFAULT_NUM_THREADS, config.getInterOpNumThreads(),
			"默认线程数 = 1");
		assertTrue(config.getLabels().isEmpty(), "默认 labels 为空");
	}

	@Test
	@DisplayName("labels 为 null 时返回空列表")
	void testLabelsNull() {
		BertIntentConfig config = BertIntentConfig.builder()
			.modelPath("model.onnx")
			.vocabPath("vocab.txt")
			.labels(null)
			.build();

		assertNotNull(config.getLabels());
		assertTrue(config.getLabels().isEmpty());
	}

	@Test
	@DisplayName("labels 不可修改（unmodifiableList）")
	void testLabelsUnmodifiable() {
		BertIntentConfig config = BertIntentConfig.builder()
			.modelPath("model.onnx")
			.vocabPath("vocab.txt")
			.labels(List.of("a", "b"))
			.build();

		assertThrows(UnsupportedOperationException.class,
			() -> config.getLabels().add("c"));
	}

	@Test
	@DisplayName("modelPath(Path) 方法")
	void testModelPathOverload() {
		BertIntentConfig config = BertIntentConfig.builder()
			.modelPath(Path.of("/tmp/bert.onnx"))
			.vocabPath("vocab.txt")
			.build();

		assertEquals(Path.of("/tmp/bert.onnx").toString(), config.getModelPath());
	}

	@Test
	@DisplayName("vocabPath(Path) 方法")
	void testVocabPathOverload() {
		BertIntentConfig config = BertIntentConfig.builder()
			.modelPath("model.onnx")
			.vocabPath(Path.of("/tmp/vocab.txt"))
			.build();

		assertEquals(Path.of("/tmp/vocab.txt").toString(), config.getVocabPath());
	}

	@Nested
	@DisplayName("参数校验")
	class Validation {

		@Test
		@DisplayName("modelPath 缺失抛异常")
		void testMissingModelPath() {
			assertThrows(IllegalArgumentException.class, () ->
				BertIntentConfig.builder()
					.vocabPath("vocab.txt")
					.build()
			);
		}

		@Test
		@DisplayName("modelPath 为空白抛异常")
		void testBlankModelPath() {
			assertThrows(IllegalArgumentException.class, () ->
				BertIntentConfig.builder()
					.modelPath("   ")
					.vocabPath("vocab.txt")
					.build()
			);
		}

		@Test
		@DisplayName("vocabPath 缺失抛异常")
		void testMissingVocabPath() {
			assertThrows(IllegalArgumentException.class, () ->
				BertIntentConfig.builder()
					.modelPath("model.onnx")
					.build()
			);
		}

		@Test
		@DisplayName("vocabPath 为空白抛异常")
		void testBlankVocabPath() {
			assertThrows(IllegalArgumentException.class, () ->
				BertIntentConfig.builder()
					.modelPath("model.onnx")
					.vocabPath("")
					.build()
			);
		}

		@Test
		@DisplayName("maxLength < 3 抛异常")
		void testMaxLengthTooSmall() {
			assertThrows(IllegalArgumentException.class, () ->
				BertIntentConfig.builder()
					.modelPath("model.onnx")
					.vocabPath("vocab.txt")
					.maxLength(2)
					.build()
			);
		}

		@Test
		@DisplayName("maxLength >= 3 正常")
		void testMaxLengthValid() {
			BertIntentConfig config = BertIntentConfig.builder()
				.modelPath("model.onnx")
				.vocabPath("vocab.txt")
				.maxLength(3)
				.build();

			assertEquals(3, config.getMaxLength());
		}
	}

	@Test
	@DisplayName("DEFAULT_MAX_LENGTH = 128")
	void testDefaultMaxLengthConstant() {
		assertEquals(128, BertIntentConfig.DEFAULT_MAX_LENGTH);
	}

	@Test
	@DisplayName("DEFAULT_NUM_THREADS = 1")
	void testDefaultNumThreadsConstant() {
		assertEquals(1, BertIntentConfig.DEFAULT_NUM_THREADS);
	}
}
