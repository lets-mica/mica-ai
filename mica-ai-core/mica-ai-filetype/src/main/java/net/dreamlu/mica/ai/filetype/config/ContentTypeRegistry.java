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
package net.dreamlu.mica.ai.filetype.config;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.filetype.model.ContentTypeInfo;
import net.dreamlu.mica.ai.filetype.model.ContentTypeLabel;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * 内容类型知识库（content_types_kb.min.json）注册表。
 *
 * <p>语义对齐 magika Python 版 {@code ContentTypeData}：kb 中缺失的字段按
 * {@code is_text} 填充默认 mime / group / description，未知 label 兜底为
 * {@code unknown} 条目。
 */
@Slf4j
public class ContentTypeRegistry {

	private static final String TEXT_PLAIN = "text/plain";
	private static final String OCTET_STREAM = "application/octet-stream";
	private static final String GROUP_UNKNOWN = "unknown";

	private final Map<String, ContentTypeInfo> types;

	public ContentTypeRegistry(Map<String, ContentTypeInfo> types) {
		this.types = types == null ? Collections.<String, ContentTypeInfo>emptyMap() : types;
	}

	public static ContentTypeRegistry load(String path) {
		if (path == null || path.isEmpty()) {
			throw new MicaAiException(
				ErrorCode.ILLEGAL_ARGUMENT, "content types 路径为空");
		}
		try {
			if (OnnxModelSession.isClasspath(path)) {
				String resource = OnnxModelSession.stripClasspathPrefix(path);
				byte[] bytes = OnnxModelSession.loadClasspathBytes(resource);
				log.info("加载 content types 知识库 (classpath): {} ({} bytes)", resource, bytes.length);
				return load(bytes);
			}
			log.info("加载 content types 知识库: {}", path);
			try (InputStream in = Files.newInputStream(Paths.get(path))) {
				return load(in);
			}
		} catch (IOException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED,
				"加载 content types 知识库失败: " + path, e);
		}
	}

	public static ContentTypeRegistry load(byte[] json) {
		try {
			Map<String, ContentTypeInfo> raw = new ObjectMapper().readValue(
				json, new TypeReference<Map<String, ContentTypeInfo>>() {
				});
			return of(raw);
		} catch (IOException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "解析 content types 知识库失败", e);
		}
	}

	public static ContentTypeRegistry load(InputStream in) throws IOException {
		Map<String, ContentTypeInfo> raw = new ObjectMapper().readValue(
			in, new TypeReference<Map<String, ContentTypeInfo>>() {
			});
		return of(raw);
	}

	private static ContentTypeRegistry of(Map<String, ContentTypeInfo> raw) {
		Map<String, ContentTypeInfo> map = new HashMap<>(Math.max(16, raw.size() * 2));
		for (Map.Entry<String, ContentTypeInfo> entry : raw.entrySet()) {
			map.put(entry.getKey(), applyDefaults(entry.getKey(), entry.getValue()));
		}
		log.debug("content types 知识库加载完成，共 {} 个条目", map.size());
		return new ContentTypeRegistry(map);
	}

	private static ContentTypeInfo applyDefaults(String label, ContentTypeInfo info) {
		if (info == null) {
			info = new ContentTypeInfo();
		}
		info.setLabel(label);
		if (info.getMimeType() == null) {
			info.setMimeType(info.isText() ? TEXT_PLAIN : OCTET_STREAM);
		}
		if (info.getGroup() == null) {
			info.setGroup(GROUP_UNKNOWN);
		}
		if (info.getDescription() == null) {
			info.setDescription(label);
		}
		if (info.getExtensions() == null) {
			info.setExtensions(Collections.<String>emptyList());
		}
		return info;
	}

	public ContentTypeInfo get(String label) {
		ContentTypeInfo info = types.get(label);
		if (info != null) {
			return info;
		}
		ContentTypeInfo unknown = types.get(ContentTypeLabel.UNKNOWN);
		if (unknown != null) {
			return unknown;
		}
		return fallbackUnknown();
	}

	private static ContentTypeInfo fallbackUnknown() {
		ContentTypeInfo info = new ContentTypeInfo();
		info.setLabel(ContentTypeLabel.UNKNOWN);
		info.setMimeType(OCTET_STREAM);
		info.setGroup(GROUP_UNKNOWN);
		info.setDescription(ContentTypeLabel.UNKNOWN);
		info.setExtensions(Collections.emptyList());
		info.setText(false);
		return info;
	}

	public int size() {
		return types.size();
	}
}
