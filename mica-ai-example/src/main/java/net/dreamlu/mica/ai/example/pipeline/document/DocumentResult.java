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
package net.dreamlu.mica.ai.example.pipeline.document;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.util.Collections;
import java.util.List;

/**
 * 智能文档流水线的处理结果。
 *
 * <p>{@code outputPng} 是处理后的整图字节；其余字段是各阶段的结构化报告，
 * 由 Controller 原样转成 JSON 返回。
 *
 * @author DreamLu
 */
@Getter
public class DocumentResult {

	/** 处理后的整图（PNG 字节） */
	private final byte[] outputPng;

	/** 输出图宽 */
	private final int width;

	/** 输出图高 */
	private final int height;

	/** 各阶段执行情况，顺序即流水线顺序 */
	private final List<StageReport> stages;

	/** 拉正阶段详情；未执行时为 null */
	private final AlignReport align;

	/** 整图方向阶段详情；未执行时为 null */
	private final OrientationReport orientation;

	/** 清晰度 / 超分阶段详情；未执行时为 null */
	private final ClarityReport clarity;

	/** 版面区域；未做版面分析时为空列表 */
	private final List<LayoutBlockReport> blocks;

	/** 文本块方向判定结果；未执行时为空列表 */
	private final List<TextBlockReport> textBlocks;

	public DocumentResult(byte[] outputPng, int width, int height,
						  List<StageReport> stages,
						  AlignReport align,
						  OrientationReport orientation,
						  ClarityReport clarity,
						  List<LayoutBlockReport> blocks,
						  List<TextBlockReport> textBlocks) {
		this.outputPng = outputPng;
		this.width = width;
		this.height = height;
		this.stages = stages == null ? Collections.<StageReport>emptyList() : stages;
		this.align = align;
		this.orientation = orientation;
		this.clarity = clarity;
		this.blocks = blocks == null ? Collections.<LayoutBlockReport>emptyList() : blocks;
		this.textBlocks = textBlocks == null ? Collections.<TextBlockReport>emptyList() : textBlocks;
	}

	/**
	 * 单个阶段的执行情况。
	 */
	@Getter
	@AllArgsConstructor
	public static class StageReport {

		/** 阶段标识：align / orientation / clarity / layout / textLine */
		private final String name;

		/** 该阶段是否真正产生了处理动作 */
		private final boolean applied;

		/** 人话描述：做了什么 / 为什么没做 */
		private final String detail;
	}

	/**
	 * 拉正阶段的角点。
	 */
	@Getter
	@AllArgsConstructor
	public static class Corner {

		/** 角点 x（原图坐标） */
		private final double x;

		/** 角点 y（原图坐标） */
		private final double y;

		/** 热力图峰值，即该角点的置信度 */
		private final double score;

		/** 角点位置：左上 / 右上 / 右下 / 左下 */
		private final String position;
	}

	/**
	 * 拉正阶段详情。
	 */
	@Getter
	@AllArgsConstructor
	public static class AlignReport {

		/** 是否执行了透视矫正 */
		private final boolean applied;

		/** 4 个角点 */
		private final List<Corner> corners;

		/** 角点最低分 */
		private final double minCornerScore;

		/** 拉正后的输出尺寸，未拉正则为原图尺寸 */
		private final int outputWidth;

		/** 拉正后的输出尺寸，未拉正则为原图尺寸 */
		private final int outputHeight;
	}

	/**
	 * 整图方向阶段详情。
	 */
	@Getter
	@AllArgsConstructor
	public static class OrientationReport {

		/** 模型标签：0 / 90 / 180 / 270 */
		private final String label;

		/** 标签置信度 */
		private final float score;

		/** 是否执行了旋转 */
		private final boolean applied;

		/** 旋转动作描述，如「逆时针 90°」 */
		private final String rotation;
	}

	/**
	 * 清晰度 / 超分阶段详情。
	 */
	@Getter
	@AllArgsConstructor
	public static class ClarityReport {

		/** 模型标签：clear / blur */
		private final String label;

		/** 标签置信度 */
		private final float score;

		/** 是否执行了超分 */
		private final boolean applied;

		/** 超分倍数，未超分为 1 */
		private final int scale;
	}

	/**
	 * 单个版面区域。
	 */
	@Getter
	@AllArgsConstructor
	public static class LayoutBlockReport {

		/** 版面类别 code，如 paragraph_title */
		private final String label;

		/** 类别置信度 */
		private final float score;

		/** 原图坐标系下的 [x1, y1, x2, y2] */
		private final int[] boundingBox;

		/** 阅读顺序序号，-1 表示不参与线性阅读序列 */
		private final int readingOrder;
	}

	/**
	 * 文本块的方向判定结果。
	 */
	@Getter
	@AllArgsConstructor
	public static class TextBlockReport {

		/** 文本块坐标 [x1, y1, x2, y2] */
		private final int[] boundingBox;

		/** 方向标签：0_degree / 180_degree */
		private final String orientation;

		/** 方向置信度 */
		private final float score;

		/** 是否被判定为倒置并已转正 */
		private final boolean flipped;
	}

}
