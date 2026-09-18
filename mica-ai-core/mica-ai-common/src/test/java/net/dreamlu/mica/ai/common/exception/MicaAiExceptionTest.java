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
package net.dreamlu.mica.ai.common.exception;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * {@link MicaAiException} 单测。
 */
class MicaAiExceptionTest {

	@Test
	void carriesCodeAndMessage() {
		MicaAiException ex = new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "boom");
		assertThat(ex.getCode()).isEqualTo(ErrorCode.MODEL_LOAD_FAILED);
		assertThat(ex.getMessage()).isEqualTo("boom");
		assertThat(ex.getCause()).isNull();
	}

	@Test
	void chainsCause() {
		IllegalStateException root = new IllegalStateException("root");
		MicaAiException ex = new MicaAiException(ErrorCode.INFERENCE_FAILED, "wrap", root);
		assertThat(ex.getCode()).isEqualTo(ErrorCode.INFERENCE_FAILED);
		assertThat(ex.getCause()).isSameAs(root);
	}
}