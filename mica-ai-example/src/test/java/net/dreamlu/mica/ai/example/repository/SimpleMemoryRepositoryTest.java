/*
 * Copyright (c) 2019-2029, Dreamlu 卢春梦 (596392912@qq.com & dreamlu.net).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.dreamlu.mica.ai.example.repository;

import net.dreamlu.mica.ai.face.model.MatchResult;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * {@link SimpleMemoryRepository} 单元测试。
 */
class SimpleMemoryRepositoryTest {

	private SimpleMemoryRepository repository;

	@BeforeEach
	void setUp() {
		repository = new SimpleMemoryRepository();
	}

	@Test
	void saveAndGetShouldRoundTripFeature() {
		float[] feature = new float[]{0.1f, 0.2f, 0.3f};
		repository.save("alice", feature);

		float[] actual = repository.getFeature("alice");
		assertNotNull(actual);
		assertEquals(feature.length, actual.length);
		for (int i = 0; i < feature.length; i++) {
			assertEquals(feature[i], actual[i]);
		}
		// 应返回克隆，避免外部修改污染仓库
		assertNotSame(feature, actual);
	}

	@Test
	void getFeatureShouldReturnNullForUnknownPerson() {
		assertNull(repository.getFeature("ghost"));
	}

	@Test
	void searchShouldReturnTopKByCosineSimilarityDescending() {
		repository.save("a", new float[]{1f, 0f, 0f});
		repository.save("b", new float[]{0f, 1f, 0f});
		repository.save("c", new float[]{1f, 1f, 0f});

		float[] query = new float[]{1f, 0f, 0f};
		List<MatchResult> top2 = repository.search(query, 2);

		assertEquals(2, top2.size());
		// a 与查询向量完全一致，相似度 1.0
		assertEquals("a", top2.get(0).getPersonId());
		assertEquals(1.0f, top2.get(0).getSimilarity(), 1e-6);
		// c 次之
		assertEquals("c", top2.get(1).getPersonId());
	}

	@Test
	void searchOnEmptyRepositoryShouldReturnEmpty() {
		List<MatchResult> result = repository.search(new float[]{1f, 0f}, 5);
		assertTrue(result.isEmpty());
	}

	@Test
	void deleteShouldRemoveFeature() {
		repository.save("alice", new float[]{1f, 0f, 0f});
		repository.delete("alice");
		assertNull(repository.getFeature("alice"));
	}

	@Test
	void deleteOnMissingShouldBeNoop() {
		repository.delete("ghost");
		// 不存在的人员不应被新建
		assertNull(repository.getFeature("ghost"));
	}
}
