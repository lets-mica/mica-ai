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

import lombok.Builder;
import lombok.Getter;

/**
 * 智能文档流水线的阶段开关。
 *
 * <p>每个阶段都可单独关闭，关闭后该阶段既不参与推理、也不出现在报告里。
 * 默认全开；{@code alignMinCornerScore} / {@code enhanceBlurThreshold}
 * 是两个阶段各自的判定门限。
 *
 * @author DreamLu
 */
@Getter
@Builder(toBuilder = true)
public class DocumentOptions {

	/** 阶段 1：透视拉正（docalign 检四角 + 拉平） */
	@Builder.Default
	private final boolean align = true;

	/**
	 * 拉正门限：4 个角点热力图峰值中的最低分。
	 * 低于此值说明画面里大概率没有成形的文档，拉正只会把图拧坏，故跳过。
	 */
	@Builder.Default
	private final double alignMinCornerScore = 0.3d;

	/** 阶段 2：整图方向矫正（0° / 90° / 180° / 270°） */
	@Builder.Default
	private final boolean orientation = true;

	/**
	 * 阶段 3：条码 / 二维码解码（纯算法零模型）。
	 *
	 * <p>放在拉正与整图转正之后、超分之前：此时几何已正、仍是原分辨率；
	 * 一维码一旦被缩放就几乎解不出，所以绝不能排在 enhance 之后。
	 */
	@Builder.Default
	private final boolean barcode = true;

	/** 阶段 4：版面分析（区域 + 阅读顺序） */
	@Builder.Default
	private final boolean layout = true;

	/** 阶段 5：文本块倒置转正 */
	@Builder.Default
	private final boolean textLine = true;

	/** 参与方向判定的文本块数量上限，避免超大文档推理过久 */
	@Builder.Default
	private final int maxTextBlocks = 20;

	/** 阶段 6：清晰度不足时超分（放最后，避免后续阶段在 4 倍图上跑） */
	@Builder.Default
	private final boolean enhance = true;

	/** 清晰度门限：clear 概率低于此值判为模糊，触发超分 */
	@Builder.Default
	private final float enhanceBlurThreshold = 0.5f;

}
