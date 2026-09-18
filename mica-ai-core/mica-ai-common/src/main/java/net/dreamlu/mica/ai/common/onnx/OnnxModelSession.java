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

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.util.IOUtil;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 单个 ONNX 模型的会话持有者（跨能力通用基础设施）。
 *
 * <p>每个模型独立加载、独立持有 {@link OrtSession}，加载失败时互不影响；
 * 资源释放（{@link #close()}）也彼此隔离，便于按需启停与单元测试。
 * classpath: 前缀的资源解析逻辑集中在此处，供各模型复用。
 *
 * <p>线程安全：{@link OrtSession} 本身线程安全，本类通过 {@code AtomicBoolean}
 * 守护 {@link #close()}，可重入；其他字段均为 final。
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
	private final AtomicBoolean closed = new AtomicBoolean(false);

	public OnnxModelSession(OrtEnvironment environment, String path,
							OrtSession.SessionOptions options, String name) {
		this.name = Objects.requireNonNull(name, "model name must not be null");
		this.environment = Objects.requireNonNull(environment, "OrtEnvironment must not be null");
		this.sourcePath = path;
		if (path == null || path.isEmpty()) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "模型路径为空: " + name);
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
			String absolutePath = Paths.get(path).toFile().getAbsolutePath();
			return environment.createSession(absolutePath, options);
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED,
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
				ErrorCode.MODEL_LOAD_FAILED,
				"无法获取 ClassLoader 以加载资源: " + resourcePath);
		}
		try (InputStream in = cl.getResourceAsStream(resourcePath)) {
			if (in == null) {
				throw new MicaAiException(
					ErrorCode.NOT_FOUND,
					"classpath 资源未找到: " + resourcePath);
			}
			return IOUtil.readAllBytes(in);
		} catch (IOException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED,
				"读取 classpath 资源失败: " + resourcePath, e);
		}
	}

	public void close() {
		if (!closed.compareAndSet(false, true)) {
			return;
		}
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
