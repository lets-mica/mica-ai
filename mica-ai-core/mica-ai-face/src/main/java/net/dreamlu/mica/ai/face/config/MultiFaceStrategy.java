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
package net.dreamlu.mica.ai.face.config;

/**
 * 1:1 比对时「单张图检出多张人脸」的选脸策略。
 *
 * <p>{@code FaceVerifier} 需要从每张图中取出<b>唯一一张</b>人脸做对齐与特征提取，
 * 本枚举决定多脸时如何取舍：
 *
 * <ul>
 *   <li>{@link #LARGEST_AREA}（默认）—— 取检测框面积最大者。现场自拍里本人通常占画面主体，
 *       是绝大多数场景的合理默认；与 1.0.0 的既有行为完全一致</li>
 *   <li>{@link #LARGEST_SCORE} —— 取检测置信度最高者。人脸被遮挡 / 侧脸导致大面积框置信度偏低时更稳</li>
 *   <li>{@link #REJECT} —— 直接拒绝比对（抛 {@code MicaAiException}，错误码
 *       {@code VERIFICATION_FAILED}）。合规要求「必须单人独照」时使用，
 *       避免算法替业务做猜测</li>
 * </ul>
 *
 * <p>注意：策略对 {@code probe} 与 {@code reference} 两张图<b>同时生效</b>。
 *
 * <p>由 {@link ModelConfig#getMultiFaceStrategy()} 配置，Starter 场景对应
 * {@code mica.ai.face.verify.strategy}。
 */
public enum MultiFaceStrategy {
	/**
	 * 取面积最大的人脸。
	 */
	LARGEST_AREA,
	/**
	 * 取检测置信度最高的人脸。
	 */
	LARGEST_SCORE,
	/**
	 * 多脸时直接拒绝比对。
	 */
	REJECT;

	/**
	 * 未显式配置时使用的默认策略，也是 {@link ModelConfig} Builder 的默认值。
	 */
	public static final MultiFaceStrategy DEFAULT = LARGEST_AREA;
}
