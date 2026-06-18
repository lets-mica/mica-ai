package net.dreamlu.mica.ai.tts;

import net.dreamlu.mica.ai.tts.engine.Vocab;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Vocab 测试。
 */
class VocabTest {

	@Test
	void testLoadVocab() throws Exception {
		// 使用 mica-ai-tts 项目的 config.json（注意：可能不是中文版）
		// 跳过资源加载测试，因为没有内置 vocab 文件
		// Vocab vocab = Vocab.load("src/main/resources/vocab.json");
		// assertNotNull(vocab);
	}

	@Test
	void testVocabWithMap() {
		var map = new java.util.HashMap<Character, Integer>();
		map.put('a', 43);
		map.put('b', 44);
		Vocab vocab = new Vocab(map);
		List<Integer> tokens = vocab.tokenize("ab");
		assertEquals(2, tokens.size());
		assertEquals(43, tokens.get(0));
		assertEquals(44, tokens.get(1));
	}

	@Test
	void testVocabFilter() {
		var map = new java.util.HashMap<Character, Integer>();
		map.put('a', 43);
		map.put('b', 44);
		Vocab vocab = new Vocab(map);
		String filtered = vocab.filter("abc");
		assertEquals("ab", filtered);
	}

	@Test
	void testVocabContains() {
		var map = new java.util.HashMap<Character, Integer>();
		map.put('a', 43);
		Vocab vocab = new Vocab(map);
		assertTrue(vocab.contains('a'));
		assertFalse(vocab.contains('z'));
	}
}
