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
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.example.pipeline.document.DocumentOptions;
import net.dreamlu.mica.ai.example.pipeline.document.DocumentPipeline;
import net.dreamlu.mica.ai.example.pipeline.document.DocumentResult;
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
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 智能文档流水线 REST 端点。
 *
 * <p>把「手机拍的文档」跑完 拉正 → 整图方向 → 版面分析 → 文本块转正 → 清晰度/超分
 * 这条链，一次调用返回结构化报告。
 *
 * @author DreamLu
 */
@Slf4j
@Tag(name = "Scenario 智能文档流水线", description = "拉正 → 整图方向 → 版面分析 → 文本块转正 → 超分")
@RestController
@RequestMapping("/scenario/document")
@RequiredArgsConstructor
public class DocumentScenarioController {

	private final DocumentPipeline pipeline;

	@Operation(summary = "智能文档处理",
		description = "串联 docalign / classify / layout / textline / enhance 五个能力；"
			+ "各阶段可用参数单独关闭。返回阶段报告、版面区域与阅读顺序；"
			+ "includeImage=true 时附带处理后整图的 base64 data URL")
	@PostMapping(value = "/process", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> process(
		@Parameter(description = "拍摄的文档图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "是否透视拉正，默认 true")
		@RequestParam(value = "align", required = false) Boolean align,
		@Parameter(description = "是否整图方向矫正，默认 true")
		@RequestParam(value = "orientation", required = false) Boolean orientation,
		@Parameter(description = "模糊时是否超分，默认 true")
		@RequestParam(value = "enhance", required = false) Boolean enhance,
		@Parameter(description = "是否版面分析，默认 true（layout 模型未启用时自动降级）")
		@RequestParam(value = "layout", required = false) Boolean layout,
		@Parameter(description = "是否转正倒置文本块，默认 true")
		@RequestParam(value = "textLine", required = false) Boolean textLine,
		@Parameter(description = "是否在报告里附带处理后整图的 base64，默认 true")
		@RequestParam(value = "includeImage", required = false) Boolean includeImage) throws IOException {
		try {
			DocumentOptions options = options(align, orientation, enhance, layout, textLine);
			DocumentResult result = pipeline.process(file.getBytes(), options);
			return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(report(result, includeImage));
		} catch (MicaAiException e) {
			log.warn("智能文档处理失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	@Operation(summary = "智能文档处理（直接出图）",
		description = "参数同上，返回处理后的 PNG 字节，可直接下载或前端展示")
	@PostMapping(value = "/preview", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<byte[]> preview(
		@Parameter(description = "拍摄的文档图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file,
		@RequestParam(value = "align", required = false) Boolean align,
		@RequestParam(value = "orientation", required = false) Boolean orientation,
		@RequestParam(value = "enhance", required = false) Boolean enhance,
		@RequestParam(value = "layout", required = false) Boolean layout,
		@RequestParam(value = "textLine", required = false) Boolean textLine) throws IOException {
		DocumentOptions options = options(align, orientation, enhance, layout, textLine);
		DocumentResult result = pipeline.process(file.getBytes(), options);
		return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).body(result.getOutputPng());
	}

	private static DocumentOptions options(Boolean align, Boolean orientation, Boolean enhance,
										   Boolean layout, Boolean textLine) {
		return DocumentOptions.builder()
			.align(align == null || align)
			.orientation(orientation == null || orientation)
			.enhance(enhance == null || enhance)
			.layout(layout == null || layout)
			.textLine(textLine == null || textLine)
			.build();
	}

	/**
	 * 把处理结果摊平成 JSON 报告。
	 */
	private static Map<String, Object> report(DocumentResult result, Boolean includeImage) {
		Map<String, Object> body = new LinkedHashMap<>();
		body.put("width", result.getWidth());
		body.put("height", result.getHeight());

		List<Map<String, Object>> stages = new ArrayList<>();
		for (DocumentResult.StageReport stage : result.getStages()) {
			Map<String, Object> item = new LinkedHashMap<>();
			item.put("name", stage.getName());
			item.put("applied", stage.isApplied());
			item.put("detail", stage.getDetail());
			stages.add(item);
		}
		body.put("stages", stages);

		DocumentResult.AlignReport align = result.getAlign();
		if (align != null) {
			Map<String, Object> alignBody = new LinkedHashMap<>();
			alignBody.put("applied", align.isApplied());
			alignBody.put("minCornerScore", align.getMinCornerScore());
			List<Map<String, Object>> corners = new ArrayList<>();
			for (DocumentResult.Corner corner : align.getCorners()) {
				Map<String, Object> item = new LinkedHashMap<>();
				item.put("position", corner.getPosition());
				item.put("x", corner.getX());
				item.put("y", corner.getY());
				item.put("score", corner.getScore());
				corners.add(item);
			}
			alignBody.put("corners", corners);
			body.put("align", alignBody);
		}

		DocumentResult.OrientationReport orientation = result.getOrientation();
		if (orientation != null) {
			Map<String, Object> orientationBody = new LinkedHashMap<>();
			orientationBody.put("label", orientation.getLabel());
			orientationBody.put("score", orientation.getScore());
			orientationBody.put("rotated", orientation.isApplied());
			orientationBody.put("rotation", orientation.getRotation());
			body.put("orientation", orientationBody);
		}

		DocumentResult.ClarityReport clarity = result.getClarity();
		if (clarity != null) {
			Map<String, Object> clarityBody = new LinkedHashMap<>();
			clarityBody.put("label", clarity.getLabel());
			clarityBody.put("score", clarity.getScore());
			clarityBody.put("enhanced", clarity.isApplied());
			clarityBody.put("scale", clarity.getScale());
			body.put("clarity", clarityBody);
		}

		List<Map<String, Object>> blocks = new ArrayList<>();
		for (DocumentResult.LayoutBlockReport block : result.getBlocks()) {
			Map<String, Object> item = new LinkedHashMap<>();
			item.put("label", block.getLabel());
			item.put("score", block.getScore());
			item.put("boundingBox", block.getBoundingBox());
			item.put("readingOrder", block.getReadingOrder());
			blocks.add(item);
		}
		body.put("blocks", blocks);

		List<Map<String, Object>> textBlocks = new ArrayList<>();
		for (DocumentResult.TextBlockReport block : result.getTextBlocks()) {
			Map<String, Object> item = new LinkedHashMap<>();
			item.put("boundingBox", block.getBoundingBox());
			item.put("orientation", block.getOrientation());
			item.put("score", block.getScore());
			item.put("flipped", block.isFlipped());
			textBlocks.add(item);
		}
		body.put("textBlocks", textBlocks);

		if (includeImage == null || includeImage) {
			byte[] png = result.getOutputPng();
			body.put("imageBase64", png == null || png.length == 0 ? null
				: "data:image/png;base64," + Base64.getEncoder().encodeToString(png));
		}
		return body;
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

}
