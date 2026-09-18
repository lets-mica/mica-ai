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
package net.dreamlu.mica.ai.common.onnx;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link OnnxModelSession} 路径解析纯函数单测 —— 不依赖 ONNX 原生库。
 */
class OnnxModelSessionPathTest {

	@Test
	void isClasspathDetectsPrefix() {
		assertThat(OnnxModelSession.isClasspath(null)).isFalse();
		assertThat(OnnxModelSession.isClasspath("")).isFalse();
		assertThat(OnnxModelSession.isClasspath("models/foo.onnx")).isFalse();
		assertThat(OnnxModelSession.isClasspath("classpath:models/foo.onnx")).isTrue();
	}

	@Test
	void stripClasspathPrefixHandlesLeadingSlash() {
		assertThat(OnnxModelSession.stripClasspathPrefix("classpath:models/foo.onnx"))
			.isEqualTo("models/foo.onnx");
		assertThat(OnnxModelSession.stripClasspathPrefix("classpath:/models/foo.onnx"))
			.isEqualTo("models/foo.onnx");
	}
}