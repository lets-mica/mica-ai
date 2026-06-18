/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.intent;

import net.dreamlu.mica.ai.intent.tokenizer.VocabLoader;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

/**
 * {@link VocabLoader} 单元测试。
 */
@DisplayName("VocabLoader 词表加载器")
class VocabLoaderTest {

	private static final ClassLoader CL = Thread.currentThread().getContextClassLoader();

	@Test
	@DisplayName("从 classpath 加载 test-vocab.txt")
	void testLoadFromClasspathResource() throws IOException {
		VocabLoader vocab = VocabLoader.loadFromClasspath(CL, "test-vocab.txt");

		assertNotNull(vocab);
		assertTrue(vocab.size() > 100, "词表应包含 100+ 个 token");
	}

	@Test
	@DisplayName("内置特殊 token ID 常量")
	void testSpecialTokenConstants() {
		assertEquals(0, VocabLoader.PAD_ID);
		assertEquals(100, VocabLoader.UNK_ID);
		assertEquals(101, VocabLoader.CLS_ID);
		assertEquals(102, VocabLoader.SEP_ID);
	}

	@Nested
	@DisplayName("特殊 token 查表")
	class SpecialTokens {

		private final VocabLoader vocab;

		SpecialTokens() throws IOException {
			vocab = VocabLoader.loadFromClasspath(CL, "test-vocab.txt");
		}

		@Test
		@DisplayName("[PAD] → ID 0")
		void testPadToken() {
			assertEquals(VocabLoader.PAD_ID, vocab.getId("[PAD]"));
			assertEquals("[PAD]", vocab.getToken(0));
		}

		@Test
		@DisplayName("[UNK] → ID 100")
		void testUnkToken() {
			assertEquals(VocabLoader.UNK_ID, vocab.getId("[UNK]"));
			assertEquals("[UNK]", vocab.getToken(100));
		}

		@Test
		@DisplayName("[CLS] → ID 101")
		void testClsToken() {
			assertEquals(VocabLoader.CLS_ID, vocab.getId("[CLS]"));
			assertEquals("[CLS]", vocab.getToken(101));
		}

		@Test
		@DisplayName("[SEP] → ID 102")
		void testSepToken() {
			assertEquals(VocabLoader.SEP_ID, vocab.getId("[SEP]"));
			assertEquals("[SEP]", vocab.getToken(102));
		}
	}

	@Nested
	@DisplayName("中文 token 查表")
	class ChineseTokens {

		private final VocabLoader vocab;

		ChineseTokens() throws IOException {
			vocab = VocabLoader.loadFromClasspath(CL, "test-vocab.txt");
		}

		@Test
		@DisplayName("常见中文单字可查到")
		void testCommonChineseChars() {
			assertTrue(vocab.contains("\u4ECA"), "vocab should contain jin");   // 今
			assertTrue(vocab.contains("\u5929"), "vocab should contain tian");   // 天
			assertTrue(vocab.contains("\u6C14"), "vocab should contain qi");     // 气
			assertTrue(vocab.contains("\u600E"), "vocab should contain zen");    // 怎
			assertTrue(vocab.contains("\u4E48"), "vocab should contain me");     // 么
			assertTrue(vocab.contains("\u6837"), "vocab should contain yang");   // 样
		}

		@Test
		@DisplayName("getToken 可反向查 ID → 中文")
		void testReverseLookup() {
			int id = vocab.getId("天");
			assertEquals("天", vocab.getToken(id));
		}

		@Test
		@DisplayName("未知 token 返回 UNK_ID")
		void testUnknownToken() {
			assertEquals(VocabLoader.UNK_ID, vocab.getId("XYZ_NOT_EXIST"));
			assertEquals(VocabLoader.UNK_ID, vocab.getId("\uFFFF"));
		}
	}

	@Nested
	@DisplayName("边界情况")
	class EdgeCases {

		private final VocabLoader vocab;

		EdgeCases() throws IOException {
			vocab = VocabLoader.loadFromClasspath(CL, "test-vocab.txt");
		}

		@Test
		@DisplayName("ID 越界 getToken 返回 [UNK]")
		void testGetTokenOutOfBounds() {
			assertEquals("[UNK]", vocab.getToken(-1));
			assertEquals("[UNK]", vocab.getToken(vocab.size()));
			assertEquals("[UNK]", vocab.getToken(Integer.MAX_VALUE));
		}

		@Test
		@DisplayName("英文/单词 token 可查到")
		void testEnglishTokens() {
			assertTrue(vocab.contains("hello"));
			assertTrue(vocab.contains("world"));
			assertTrue(vocab.contains("weather"));
			assertTrue(vocab.contains("music"));
		}

		@Test
		@DisplayName("size() 返回正确数量")
		void testSize() {
			assertTrue(vocab.size() > 200, "测试词表应有 200+ 条目");
		}
	}

	@Test
	@DisplayName("从 classpath 加载（ClassLoader）")
	void testLoadFromClasspath() throws IOException {
		ClassLoader cl = Thread.currentThread().getContextClassLoader();
		VocabLoader vocab = VocabLoader.loadFromClasspath(cl, "test-vocab.txt");

		assertNotNull(vocab);
		assertTrue(vocab.size() > 100);
		assertTrue(vocab.contains("[CLS]"));
	}

	@Test
	@DisplayName("加载不存在的文件抛 IOException")
	void testLoadMissingFile() {
		assertThrows(IOException.class, () ->
			VocabLoader.load(Path.of("nonexistent_vocab_xyz.txt"))
		);
	}
}
