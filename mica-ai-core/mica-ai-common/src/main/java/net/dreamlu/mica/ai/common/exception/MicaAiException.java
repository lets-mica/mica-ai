/*
 * Copyright (c) 2024-2026 mica-ai
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
