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
import net.dreamlu.mica.ai.layout.model.LayoutResult;
import net.dreamlu.mica.ai.layout.pipeline.LayoutPipeline;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 文档版面分析 REST 端点（PP-DocLayoutV3，25 类 + 阅读顺序）。
 */
@Tag(name = "Layout 文档版面分析", description = "PP-DocLayoutV3 · 25 类版面区域 + 阅读顺序")
@RestController
@RequestMapping("/layout")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.layout", name = "enabled", havingValue = "true")
public class LayoutController {

	private final LayoutPipeline pipeline;

	private static Map<String, Object> view(LayoutResult r) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("label", r.getLabel() == null ? null : r.getLabel().name());
		m.put("labelCode", r.getLabelCode());
		m.put("score", r.getScore());
		m.put("boundingBox", r.getBoundingBox());
		m.put("readingOrder", r.getReadingOrder());
		return m;
	}

	@Operation(summary = "上传图片版面分析", description = "对上传的文档图片做版面分析，返回 25 类区域 + 阅读顺序（readingOrder 升序）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "分析成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/detect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public List<Map<String, Object>> detectUpload(
		@Parameter(description = "文档图片（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		return pipeline.detectBytes(file.getBytes()).stream()
			.map(LayoutController::view)
			.collect(Collectors.toList());
	}

	@Operation(summary = "原始字节版面分析", description = "上传图片二进制字节数组做版面分析（适合集成方直接传 Buffer / byte[]）")
	@PostMapping(value = "/detect-bytes", consumes = MediaType.APPLICATION_OCTET_STREAM_VALUE)
	public List<Map<String, Object>> detectBytes(
		@Parameter(description = "图片二进制内容", required = true,
			content = @Content(mediaType = MediaType.APPLICATION_OCTET_STREAM_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestParam("file") MultipartFile file) throws IOException {
		return pipeline.detectBytes(file.getBytes()).stream()
			.map(LayoutController::view)
			.collect(Collectors.toList());
	}
}
