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

import lombok.Getter;

/**
 * mica-ai 统一异常。
 *
 * <p>包装底层 ONNX Runtime / OpenCV 抛出的异常，避免业务层直接依赖第三方异常类型。
 */
@Getter
public class MicaAiException extends RuntimeException {

	private static final long serialVersionUID = 1L;

	/**
	 * 错误码
	 */
	private final ErrorCode code;

	public MicaAiException(ErrorCode code, String message) {
		super(message);
		this.code = code;
	}

	public MicaAiException(ErrorCode code, String message, Throwable cause) {
		super(message, cause);
		this.code = code;
	}
}
