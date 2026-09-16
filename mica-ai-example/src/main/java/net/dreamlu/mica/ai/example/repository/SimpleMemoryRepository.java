/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example.repository;

import net.dreamlu.mica.ai.face.model.MatchResult;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 基于 {@link ConcurrentHashMap} 的内存版向量仓库示例实现。
 * <p>
 * 使用余弦相似度进行暴力搜索，仅供 Demo 与单机测试使用。
 * 生产环境请自行替换为 Milvus / Qdrant / Elasticsearch / pgvector 等专用向量库。
 * </p>
 */
public class SimpleMemoryRepository implements VectorRepository {

	private final Map<String, float[]> store = new ConcurrentHashMap<>();

	private static float cosine(float[] a, float[] b) {
		if (a.length != b.length) {
			return 0f;
		}
		double dot = 0d;
		double na = 0d;
		double nb = 0d;
		for (int i = 0; i < a.length; i++) {
			dot += (double) a[i] * b[i];
			na += (double) a[i] * a[i];
			nb += (double) b[i] * b[i];
		}
		double denom = Math.sqrt(na) * Math.sqrt(nb);
		if (denom < 1e-10) {
			return 0f;
		}
		return (float) (dot / denom);
	}

	@Override
	public void save(String personId, float[] feature) {
		if (personId == null || feature == null) {
			return;
		}
		store.put(personId, feature.clone());
	}

	@Override
	public List<MatchResult> search(float[] query, int topK) {
		if (query == null || query.length == 0 || store.isEmpty()) {
			return Collections.emptyList();
		}
		List<MatchResult> results = new ArrayList<>(store.size());
		for (Map.Entry<String, float[]> entry : store.entrySet()) {
			float sim = cosine(query, entry.getValue());
			results.add(new MatchResult(entry.getKey(), sim));
		}
		results.sort((a, b) -> Float.compare(b.getSimilarity(), a.getSimilarity()));
		if (results.size() > topK) {
			return new ArrayList<>(results.subList(0, Math.max(topK, 0)));
		}
		return results;
	}

	@Override
	public float[] getFeature(String personId) {
		if (personId == null) {
			return null;
		}
		float[] f = store.get(personId);
		return f == null ? null : f.clone();
	}

	@Override
	public void delete(String personId) {
		if (personId != null) {
			store.remove(personId);
		}
	}
}
