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
import net.dreamlu.mica.ai.common.util.IOUtil;
import net.dreamlu.mica.ai.plate.model.PlateResult;
import net.dreamlu.mica.ai.plate.pipeline.PlatePipeline;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 车牌识别 REST 端点（HyperLPR3 v20230229 Java 复刻）。
 */
@Tag(name = "Plate 车牌识别", description = "HyperLPR3 · 检测 + 128d 无 · CRNN 识别 + 颜色分类")
@RestController
@RequestMapping("/plate")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.plate", name = "enabled", havingValue = "true", matchIfMissing = true)
public class PlateController {
	private final PlatePipeline pipeline;

	private static List<Map<String, Object>> view(List<PlateResult> results) {
		List<Map<String, Object>> list = new ArrayList<>();
		for (PlateResult r : results) {
			Map<String, Object> m = new LinkedHashMap<>();
			m.put("plateCode", r.getPlateCode());
			m.put("plateType", r.getPlateType() == null ? null : r.getPlateType().getCode());
			m.put("detectionConfidence", r.getDetectionConfidence());
			m.put("recognitionConfidence", r.getRecognitionConfidence());
			m.put("boundingBox", r.getBoundingBox());
			m.put("landmarks", r.getLandmarks());
			list.add(m);
		}
		return list;
	}

	@Operation(summary = "上传图片识别车牌", description = "检测车牌 + 透视校正 + CRNN 识别 + 颜色分类，返回车牌号 / 类型 / 置信度 / 框 / 关键点")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "识别成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public List<Map<String, Object>> recognize(
		@Parameter(description = "待识别图片（jpg / png）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try (InputStream in = file.getInputStream()) {
			return view(pipeline.recognizeBytes(IOUtil.readAllBytes(in)));
		}
	}
}
