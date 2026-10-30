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
package net.dreamlu.mica.ai.example.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.classify.core.ClassifyEngine;
import net.dreamlu.mica.ai.classify.model.ClassifyResult;
import net.dreamlu.mica.ai.classify.model.ClassifyTopK;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.example.config.ClassifyModelRegistry;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * PULC 通用图像分类 REST 端点。
 * <p>
 * 默认模型取 starter 配置（{@code mica.ai.classify.*}）；带 {@code model} 参数时
 * 走 {@code mica.example.classify.models} 下声明的其它 PULC 模型
 * （整图方向 / 清晰度 / 人物存在 / 表格 6 属性）。
 * </p>
 */
@Slf4j
@Tag(name = "Classify 通用图像分类", description = "PaddleClas PULC · 整图方向 / 清晰度 / 人物存在 / 表格属性")
@RestController
@RequestMapping("/classify")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.classify", name = "enabled", havingValue = "true", matchIfMissing = true)
public class ClassifyController {

	/**
	 * starter 装配的默认模型引擎（{@code mica.ai.classify.*}）。
	 */
	private final ClassifyEngine classifyEngine;

	/**
	 * 额外 PULC 模型的懒加载注册表。
	 */
	private final ClassifyModelRegistry registry;

	private static ResponseEntity<Object> json(Map<String, Object> body) {
		return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

	/**
	 * 分类：返回 Top-K 与全量概率。
	 */
	@Operation(summary = "通用图像分类",
		description = "model 不传时用 starter 默认模型（mica.ai.classify）；"
			+ "传 model 可切换到 mica.example.classify.models 下声明的其它 PULC 模型。"
			+ "返回 topK（按概率降序）、scores（按标签索引的全量概率）、"
			+ "多标签模型另返回 hits（概率 ≥ 0.5 的属性）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "分类成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败 / 模型名未知", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/classify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> classify(
		@Parameter(description = "待分类图片（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "模型名，不传则用 starter 默认模型；"
			+ "可选值见 mica.example.classify.models")
		@RequestParam(value = "model", required = false) String model) throws IOException {
		try {
			ClassifyEngine engine = classifyEngine;
			String modelName = classifyEngine.getConfig().getModelVersion();
			if (model != null && !model.trim().isEmpty()) {
				engine = registry.get(model.trim());
				modelName = model.trim();
			}
			ClassifyResult result = engine.classifyBytes(file.getBytes());
			if (result == null) {
				return error("图像为空或无法解码", HttpStatus.BAD_REQUEST);
			}

			Map<String, Object> body = new LinkedHashMap<>();
			body.put("model", modelName);
			body.put("multiLabel", result.isMultiLabel());
			body.put("classCount", result.classCount());
			body.put("topLabel", result.topLabel());
			body.put("topScore", result.topScore());

			List<Map<String, Object>> topK = new java.util.ArrayList<>();
			for (ClassifyTopK item : result.getTopK()) {
				Map<String, Object> m = new LinkedHashMap<>();
				m.put("index", item.getIndex());
				m.put("label", item.getLabel());
				m.put("score", item.getScore());
				topK.add(m);
			}
			body.put("topK", topK);

			Map<String, Object> scores = new LinkedHashMap<>();
			String[] labels = engine.labels();
			for (int i = 0; i < result.classCount(); i++) {
				String label = i < labels.length ? labels[i] : String.valueOf(i);
				scores.put(label, result.scoreAt(i));
			}
			body.put("scores", scores);

			if (result.isMultiLabel()) {
				List<Map<String, Object>> hits = new java.util.ArrayList<>();
				for (ClassifyTopK item : result.labelsAbove(0.5f)) {
					Map<String, Object> m = new LinkedHashMap<>();
					m.put("index", item.getIndex());
					m.put("label", item.getLabel());
					m.put("score", item.getScore());
					hits.add(m);
				}
				body.put("hits", hits);
			}
			return json(body);
		} catch (MicaAiException e) {
			log.warn("图像分类失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}
}
