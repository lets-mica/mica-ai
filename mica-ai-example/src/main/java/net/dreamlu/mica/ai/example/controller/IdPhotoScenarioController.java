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
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoBackground;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoMaskSource;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoOptions;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoPipeline;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoResult;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoSpec;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.IOException;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 证件照一键生成 REST 端点。
 *
 * <p>把人像照片跑完 检人脸 → 质量评估 → 抠像换底 → 按规格裁剪 → 超分 这条链，
 * 一次调用产出符合规格的证件照。
 *
 * <p>质量评估始终执行、分数始终回填；是否因质量不达标而拒绝出片由
 * {@code qualityGate} 参数决定（默认 false，即放行）。
 *
 * @author DreamLu
 */
@Slf4j
@Tag(name = "Scenario 证件照一键生成", description = "检人脸 → 质量评估 → 抠像换底 → 按规格裁剪 → 超分")
@RestController
@RequestMapping("/scenario/idphoto")
@RequiredArgsConstructor
public class IdPhotoScenarioController {

	private final IdPhotoPipeline pipeline;

	@Operation(summary = "生成证件照",
		description = "spec 取 SMALL_ONE / ONE / LARGE_ONE / SMALL_TWO / TWO / LARGE_TWO；"
			+ "background 取 WHITE / BLUE / RED；maskSource 取 MATTING / HUMANSEG。"
			+ "返回质量分与是否达标、人脸框、裁剪矩形与成片尺寸；includeImage=true 时附带成片 base64")
	@PostMapping(value = "/generate", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> generate(
		@Parameter(description = "人像照片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "尺寸规格，默认 ONE（一寸）")
		@RequestParam(value = "spec", required = false) String spec,
		@Parameter(description = "底色，默认 WHITE（白底）")
		@RequestParam(value = "background", required = false) String background,
		@Parameter(description = "目标 DPI，默认 300")
		@RequestParam(value = "dpi", required = false) Integer dpi,
		@Parameter(description = "掩码来源：MATTING（默认）/ HUMANSEG")
		@RequestParam(value = "maskSource", required = false) String maskSource,
		@Parameter(description = "是否做质量门控：true=质量不达标直接拒收不出片，false=放行。默认 false")
		@RequestParam(value = "qualityGate", required = false) Boolean qualityGate,
		@Parameter(description = "质量门限，默认 0.5")
		@RequestParam(value = "minQuality", required = false) Float minQuality,
		@Parameter(description = "是否超分，默认 true")
		@RequestParam(value = "upscale", required = false) Boolean upscale,
		@Parameter(description = "是否附带成片 base64，默认 true")
		@RequestParam(value = "includeImage", required = false) Boolean includeImage) throws IOException {
		try {
			IdPhotoOptions options = options(spec, background, dpi, maskSource,
				qualityGate, minQuality, upscale);
			IdPhotoResult result = pipeline.process(file.getBytes(), options);
			return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(report(result, includeImage));
		} catch (MicaAiException e) {
			log.warn("证件照生成失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	@Operation(summary = "生成证件照（直接出图）",
		description = "参数同上，返回成片 PNG 字节；开启质量门控且质量不达标时返回 422 与错误 JSON")
	@PostMapping(value = "/preview", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<byte[]> preview(
		@Parameter(description = "人像照片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file,
		@RequestParam(value = "spec", required = false) String spec,
		@RequestParam(value = "background", required = false) String background,
		@RequestParam(value = "dpi", required = false) Integer dpi,
		@RequestParam(value = "maskSource", required = false) String maskSource,
		@RequestParam(value = "qualityGate", required = false) Boolean qualityGate,
		@RequestParam(value = "minQuality", required = false) Float minQuality,
		@RequestParam(value = "upscale", required = false) Boolean upscale) throws IOException {
		IdPhotoOptions options = options(spec, background, dpi, maskSource,
			qualityGate, minQuality, upscale);
		IdPhotoResult result = pipeline.process(file.getBytes(), options);
		if (!result.isPassed()) {
			throw new ResponseStatusException(HttpStatus.UNPROCESSABLE_ENTITY,
				result.getReason() + ": " + result.getHint());
		}
		return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).body(result.getOutputPng());
	}

	/**
	 * 列出可选的规格与底色，方便前端渲染下拉框。
	 */
	@Operation(summary = "列出可选规格与底色")
	@org.springframework.web.bind.annotation.GetMapping("/options")
	public Map<String, Object> options() {
		Map<String, Object> body = new LinkedHashMap<>();
		List<Map<String, Object>> specs = Arrays.stream(IdPhotoSpec.values())
			.map(spec -> {
				Map<String, Object> item = new LinkedHashMap<>();
				item.put("name", spec.name());
				item.put("label", spec.getLabel());
				item.put("widthMm", spec.getWidthMm());
				item.put("heightMm", spec.getHeightMm());
				item.put("widthPx300", spec.widthPx(300));
				item.put("heightPx300", spec.heightPx(300));
				return item;
			}).collect(Collectors.toList());
		body.put("specs", specs);
		List<Map<String, Object>> backgrounds = Arrays.stream(IdPhotoBackground.values())
			.map(bg -> {
				Map<String, Object> item = new LinkedHashMap<>();
				item.put("name", bg.name());
				item.put("label", bg.getLabel());
				item.put("rgb", bg.rgb());
				return item;
			}).collect(Collectors.toList());
		body.put("backgrounds", backgrounds);
		body.put("maskSources", Arrays.stream(IdPhotoMaskSource.values())
			.map(ms -> {
				Map<String, Object> item = new LinkedHashMap<>();
				item.put("name", ms.name());
				item.put("label", ms.getLabel());
				item.put("module", ms.getModule());
				return item;
			}).collect(Collectors.toList()));
		return body;
	}

	private static IdPhotoOptions options(String spec, String background, Integer dpi,
										   String maskSource, Boolean qualityGate,
										   Float minQuality, Boolean upscale) {
		IdPhotoOptions.IdPhotoOptionsBuilder builder = IdPhotoOptions.builder();
		IdPhotoSpec specValue = IdPhotoSpec.of(spec);
		if (specValue != null) {
			builder.spec(specValue);
		}
		IdPhotoBackground backgroundValue = IdPhotoBackground.of(background);
		if (backgroundValue != null) {
			builder.background(backgroundValue);
		}
		IdPhotoMaskSource maskSourceValue = IdPhotoMaskSource.of(maskSource);
		if (maskSourceValue != null) {
			builder.maskSource(maskSourceValue);
		}
		if (dpi != null && dpi > 0) {
			builder.dpi(dpi);
		}
		if (qualityGate != null) {
			builder.qualityGate(qualityGate);
		}
		if (minQuality != null) {
			builder.minQuality(minQuality);
		}
		if (upscale != null) {
			builder.upscale(upscale);
		}
		return builder.build();
	}

	private static Map<String, Object> report(IdPhotoResult result, Boolean includeImage) {
		Map<String, Object> body = new LinkedHashMap<>();
		body.put("passed", result.isPassed());
		body.put("reason", result.getReason());
		body.put("hint", result.getHint());
		body.put("spec", result.getSpec());
		body.put("background", result.getBackground());
		body.put("maskSource", result.getMaskSource());
		body.put("qualityScore", result.getQualityScore());
		body.put("qualityPassed", result.getQualityPassed());
		body.put("faceScore", result.getFaceScore());
		body.put("faceBox", result.getFaceBox());
		body.put("cropBox", result.getCropBox());
		body.put("width", result.getWidth());
		body.put("height", result.getHeight());
		body.put("upscaled", result.isUpscaled());
		body.put("upscaleScale", result.getUpscaleScale());
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
