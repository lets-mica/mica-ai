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
package net.dreamlu.mica.ai.matting.model;

import lombok.Getter;
import org.opencv.core.Mat;

/**
 * 抠图结果：原尺寸 alpha 掩码。
 *
 * <p>{@link #alpha} 是 {@code CV_32FC1} 的 {@code Mat}，尺寸与输入原图一致，
 * 取值已归一到 {@code [0,1]}（1 = 前景，0 = 背景）。
 *
 * <p>本类实现 {@link AutoCloseable}，{@link #close()} 会释放底层
 * {@link Mat}；推荐 try-with-resources：
 *
 * <pre>{@code
 * try (MattingResult r = engine.matteBytes(bytes)) {
 *     Mat alpha = r.getAlpha();
 * }
 * }</pre>
 */
@Getter
public class MattingResult implements AutoCloseable {

	private final Mat alpha;
	private final int width;
	private final int height;

	public MattingResult(Mat alpha, int width, int height) {
		this.alpha = alpha;
		this.width = width;
		this.height = height;
	}

	/**
	 * 释放底层 alpha {@link Mat}，可重复调用。
	 */
	@Override
	public void close() {
		if (alpha != null && !alpha.empty()) {
			alpha.release();
		}
	}
}
