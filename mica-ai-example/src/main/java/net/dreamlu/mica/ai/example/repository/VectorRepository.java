/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example.repository;

import net.dreamlu.mica.ai.face.model.MatchResult;

import java.util.List;

/**
 * 示例模块内的向量仓 SPI。
 * <p>
 * mica-ai-face 不提供向量检索实现；用户可在自己的业务模块里定义接口并实现，
 * 然后把 {@code search} 当成 {@code (query, topK) -> matches} 回调传给 {@link net.dreamlu.mica.ai.example.pipeline.FacePipeline}。
 * </p>
 */
public interface VectorRepository {

	/**
	 * 注册或更新一个人员的特征向量
	 *
	 * @param personId 人员 ID
	 * @param feature  人脸特征向量（SFace 为 128 维，已 L2 归一化）
	 */
	void save(String personId, float[] feature);

	/**
	 * 检索与查询向量最相似的 TopK 条记录
	 *
	 * @param query 查询向量
	 * @param topK  返回的最大结果数
	 * @return 按相似度降序排列的匹配结果
	 */
	List<MatchResult> search(float[] query, int topK);

	/**
	 * 获取指定人员的特征向量（用于 1:1 比对场景）。
	 *
	 * @param personId 人员 ID
	 * @return 特征向量，未注册则返回 {@code null}
	 */
	float[] getFeature(String personId);

	/**
	 * 删除指定人员的特征向量
	 *
	 * @param personId 人员 ID
	 */
	void delete(String personId);
}
