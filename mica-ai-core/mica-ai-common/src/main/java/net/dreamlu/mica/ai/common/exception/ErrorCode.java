/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.exception;

/**
 * 错误码枚举。
 *
 * <p>本枚举是 mica-ai 跨能力的统一错误码分类；
 * 各能力内部的细分错误码通过 message 透传。
 */
public enum ErrorCode {
	/**
	 * 模型加载失败
	 */
	MODEL_LOAD_FAILED,
	/**
	 * 人脸检测失败
	 */
	DETECTION_FAILED,
	/**
	 * 人脸对齐失败
	 */
	ALIGNMENT_FAILED,
	/**
	 * 特征提取失败
	 */
	EXTRACTION_FAILED,
	/**
	 * 活体检测失败
	 */
	LIVENESS_FAILED,
	/**
	 * 人脸 1:1 比对失败
	 */
	VERIFICATION_FAILED,
	/**
	 * 人脸头像提取失败
	 */
	AVATAR_FAILED,
	/**
	 * 证件卡片提取失败
	 */
	CARD_FAILED,
	/**
	 * 图片编码失败
	 */
	ENCODE_FAILED,
	/**
	 * 推理失败（ONNX / OpenCV 调用抛出）
	 */
	INFERENCE_FAILED,
	/**
	 * 输入参数非法
	 */
	ILLEGAL_ARGUMENT,
	/**
	 * 资源未找到（路径不存在、classpath 资源缺失等）
	 */
	NOT_FOUND,
	/**
	 * 其它未分类错误
	 */
	UNKNOWN
}
