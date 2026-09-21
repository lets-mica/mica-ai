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
package net.dreamlu.mica.ai.matting.config;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;

import java.util.Locale;

/**
 * 掩码输出节点选择策略：从模型的多个输出中定位「融合显著性掩码」（d0）。
 *
 * <p>U²-Net 族导出恒有 7 个同形输出（d0..d6），其中 <b>d0 是唯一可直接作为 alpha 使用的节点</b>，
 * d1..d6 是训练期 deep supervision 的中间输出。但不同导出方式下 d0 的命名与顺序可能不同：
 * <ul>
 *   <li>rembg 的 torch.onnx.export 产物：输出名是数字（实测 {@code 1959..1965}），<b>无稳定语义</b>，
 *       d0 恰好是第 0 个</li>
 *   <li>部分转换脚本会把输出重命名为 {@code d0..d6} 或 {@code alpha} / {@code saliency} / {@code mask}</li>
 * </ul>
 *
 * <p>因此本枚举把「怎么找 d0」这件事显式化为一个可配置策略，而不是散落在检测器代码里的
 * if-else。默认 {@link #AUTO} 覆盖上述全部情况。
 */
@Getter
@RequiredArgsConstructor
public enum MattingOutputSelect {

	/**
	 * 自动：先按名称提示（{@code d0} / {@code alpha} / {@code saliency} / {@code mask}）精确匹配，
	 * 命中则用；否则要求输出个数等于 {@link #expectedOutputs 期望值} 并取<b>第 0 个</b>
	 * float {@code [1,1,H,W]} 输出。这是 rembg 导出族的正确解法。
	 */
	AUTO("自动：按名称提示，否则取首个单通道输出", 7),

	/**
	 * 强制取第 0 个输出，跳过名称匹配。
	 *
	 * <p>适用于「已知首个输出即 d0，但节点名恰好撞上名称提示」的场景——例如某些导出把
	 * deep supervision 输出命名为 {@code d1_score}，若走 {@link #AUTO} 的模糊子串匹配可能误命中。
	 * 显式指定本策略可消除这类歧义。
	 */
	FIRST("强制取第 0 个输出", 7),

	/**
	 * 强制按名称精确匹配 {@code d0}（仅精确名，不做子串兜底），匹配不到即快速失败。
	 *
	 * <p>适用于对导出产物有严格约定的生产环境：宁可启动失败，也不接受「碰巧取对」。
	 */
	D0("强制精确匹配名为 d0 的输出", 7);

	/**
	 * 本模块默认策略。
	 */
	public static final MattingOutputSelect DEFAULT = AUTO;

	private final String description;

	/**
	 * 该策略下的期望输出个数；U²-Net 族恒为 7。
	 */
	private final int expectedOutputs;

	/**
	 * 校验枚举与模型实际输出是否自洽。
	 *
	 * @param actualOutputs 模型实际输出个数
	 * @throws MicaAiException 输出个数与该策略期望不符时抛出，错误码 {@link ErrorCode#ILLEGAL_ARGUMENT}
	 */
	public void validate(int actualOutputs) {
		if (actualOutputs != expectedOutputs) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"输出选择策略 " + name() + " 期望 " + expectedOutputs
					+ " 个输出，模型实际为 " + actualOutputs
					+ "；请确认模型是否属于 U²-Net 族");
		}
	}

	/**
	 * 按名称提示解析输出节点名。
	 *
	 * @param outputNames 模型输出节点名（原始大小写）
	 * @return 命中的节点名；{@link #D0} 仅精确匹配，{@link #FIRST} 恒为 null（交由调用方取首个）
	 * @throws MicaAiException {@link #D0} 策略下未匹配到 {@code d0} 时抛出
	 */
	public String resolveByName(String[] outputNames) {
		if (this == FIRST) {
			return null;
		}
		if (this == D0) {
			for (String name : outputNames) {
				if ("d0".equals(name.toLowerCase(Locale.ROOT))) {
					return name;
				}
			}
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"输出选择策略 D0 未在模型中找到名为 d0 的输出节点: " + String.join(", ", outputNames));
		}
		String hinted = exactMatch(outputNames, "d0", "alpha", "saliency", "mask");
		if (hinted != null) {
			return hinted;
		}
		return fuzzyMatch(outputNames, "d0", "alpha", "saliency", "mask");
	}

	private static String exactMatch(String[] names, String... hints) {
		for (String hint : hints) {
			for (String name : names) {
				if (name.equalsIgnoreCase(hint)) {
					return name;
				}
			}
		}
		return null;
	}

	private static String fuzzyMatch(String[] names, String... hints) {
		for (String hint : hints) {
			String lowerHint = hint.toLowerCase(Locale.ROOT);
			for (String name : names) {
				if (name.toLowerCase(Locale.ROOT).contains(lowerHint)) {
					return name;
				}
			}
		}
		return null;
	}
}
