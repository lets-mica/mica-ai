/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.plate.model;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.Locale;

/**
 * 车牌类型，对齐 HyperLPR3 {@code common/typedef.py}，{@link #getValue()} 与 Python 常量值一致。
 */
@Getter
@RequiredArgsConstructor
public enum PlateType {

	/**
	 * 未知（Python: UNKNOWN = -1，交给颜色分类器兜底）
	 */
	UNKNOWN(-1),
	/**
	 * 蓝牌（Python: BLUE = 0）
	 */
	BLUE(0),
	/**
	 * 黄牌单层（Python: YELLOW_SINGLE = 1）
	 */
	YELLOW_SINGLE(1),
	/**
	 * 白牌（警牌 / WJ，Python: WHILE_SINGLE = 2）
	 */
	WHITE_POLICE(2),
	/**
	 * 绿牌新能源（Python: GREEN = 3）
	 */
	GREEN(3),
	/**
	 * 黑牌港澳（Python: BLACK_HK_MACAO = 4）
	 */
	HONG_KONG_MACAO(4),
	/**
	 * 香港单层（Python: HK_SINGLE = 5）
	 */
	HK_SINGLE(5),
	/**
	 * 香港双层（Python: HK_DOUBLE = 6）
	 */
	HK_DOUBLE(6),
	/**
	 * 澳门单层（Python: MACAO_SINGLE = 7）
	 */
	MACAO_SINGLE(7),
	/**
	 * 澳门双层（Python: MACAO_DOUBLE = 8）
	 */
	MACAO_DOUBLE(8),
	/**
	 * 黄牌双层（Python: YELLOW_DOUBLE = 9）
	 */
	YELLOW_DOUBLE(9);

	/**
	 * -- GETTER --
	 * 对齐 Python
	 * 的常量值（UNKNOWN 为 -1，其余 0-9）。
	 */
	private final int value;

	/**
	 * 序列化名（小写下划线），供 JSON / 配置使用。
	 */
	public String getCode() {
		return name().toLowerCase(Locale.ROOT);
	}
}
