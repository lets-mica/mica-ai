/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.onnx;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.MicaAiException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Objects;

/**
 * 单个 ONNX 模型的会话持有者（跨能力通用基础设施）。
 *
 * <p>每个模型独立加载、独立持有 {@link OrtSession}，加载失败时互不影响；
 * 资源释放（{@link #close()}）也彼此隔离，便于按需启停与单元测试。
 * classpath: 前缀的资源解析逻辑集中在此处，供各模型复用。
 *
 * @author L.cm
 */
@Slf4j
@Getter
public class OnnxModelSession {

	/**
	 * classpath: 前缀，Spring 配置常见用法，ORT 不能直接解析
	 */
	public static final String CLASSPATH_PREFIX = "classpath:";

	private final String name;
	private final OrtEnvironment environment;
	private final OrtSession session;
	private final String sourcePath;

	public OnnxModelSession(OrtEnvironment environment, String path,
							OrtSession.SessionOptions options, String name) {
		this.name = Objects.requireNonNull(name, "model name must not be null");
		this.environment = Objects.requireNonNull(environment, "OrtEnvironment must not be null");
		this.sourcePath = path;
		if (path == null || path.isEmpty()) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.MODEL_LOAD_FAILED, "模型路径为空: " + name);
		}
		this.session = createSession(path, options, name);
	}

	private OrtSession createSession(String path, OrtSession.SessionOptions options, String name) {
		try {
			if (isClasspath(path)) {
				String resource = stripClasspathPrefix(path);
				byte[] bytes = loadClasspathBytes(resource);
				log.info("加载 {} 模型 (classpath): {} ({} bytes)", name, resource, bytes.length);
				return environment.createSession(bytes, options);
			}
			log.info("加载 {} 模型: {}", name, path);
			return environment.createSession(path, options);
		} catch (OrtException e) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.MODEL_LOAD_FAILED,
				"加载 " + name + " 模型失败: " + path, e);
		}
	}

	public static boolean isClasspath(String path) {
		return path != null && path.startsWith(CLASSPATH_PREFIX);
	}

	public static String stripClasspathPrefix(String path) {
		String resource = path.substring(CLASSPATH_PREFIX.length());
		if (resource.startsWith("/")) {
			resource = resource.substring(1);
		}
		return resource;
	}

	public static byte[] loadClasspathBytes(String resourcePath) {
		ClassLoader cl = Thread.currentThread().getContextClassLoader();
		if (cl == null) {
			cl = OnnxModelSession.class.getClassLoader();
		}
		if (cl == null) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.MODEL_LOAD_FAILED,
				"无法获取 ClassLoader 以加载资源: " + resourcePath);
		}
		try (InputStream in = cl.getResourceAsStream(resourcePath)) {
			if (in == null) {
				throw new MicaAiException(
					MicaAiException.ErrorCode.NOT_FOUND,
					"classpath 资源未找到: " + resourcePath);
			}
			return readAllBytes(in);
		} catch (IOException e) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.MODEL_LOAD_FAILED,
				"读取 classpath 资源失败: " + resourcePath, e);
		}
	}

	private static byte[] readAllBytes(InputStream in) throws IOException {
		ByteArrayOutputStream out = new ByteArrayOutputStream(Math.max(64, in.available()));
		byte[] buf = new byte[8192];
		int n;
		while ((n = in.read(buf)) != -1) {
			out.write(buf, 0, n);
		}
		return out.toByteArray();
	}

	public void close() {
		if (session == null) {
			return;
		}
		try {
			session.close();
		} catch (Exception e) {
			log.warn("关闭 {} OrtSession 失败: {}", name, e.getMessage());
		}
	}
}