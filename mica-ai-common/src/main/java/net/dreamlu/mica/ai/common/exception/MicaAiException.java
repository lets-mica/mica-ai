/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.exception;

/**
 * mica-ai 统一异常。
 *
 * <p>所有 mica-ai 模块的运行时异常均继承此类，
 * 方便调用方统一捕获和处理。
 */
public class MicaAiException extends RuntimeException {

	public MicaAiException(String message) {
		super(message);
	}

	public MicaAiException(String message, Throwable cause) {
		super(message, cause);
	}

	public MicaAiException(Throwable cause) {
		super(cause);
	}
}
