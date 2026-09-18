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
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.avatar.AvatarExtractor;
import net.dreamlu.mica.ai.face.avatar.AvatarOptions;
import net.dreamlu.mica.ai.face.avatar.AvatarResult;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;
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
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 人脸头像提取端点。
 * <p>
 * 从图片或证件中裁出规范化的人脸头像：正方形、固定尺寸、自动留白，可按眼线摆正侧倾的人脸，
 * 并在必要时自动摆正 90° 级大角度朝向、对「大图小脸」启用分块检测。
 * 所有查询参数均为可选，未传时使用 {@code mica.ai.face.avatar.*} 配置的默认值；
 * 参数语义详见 {@link AvatarOptions}。
 * </p>
 */
@Slf4j
@Tag(name = "Face 头像提取", description = "从图片 / 证件中裁出规范化人脸头像")
@RestController
@RequestMapping("/face/avatar")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.face", name = "enabled", havingValue = "true", matchIfMissing = true)
public class AvatarController {

	private final AvatarExtractor avatarExtractor;

	/**
	 * 提取头像：返回画面中面积最大的人脸，直接输出图片字节。
	 */
	@Operation(summary = "提取头像（最大人脸）",
		description = "返回画面中面积最大的人脸头像，Content-Type 由 format 决定；"
			+ "faceScale 1.3 约等于证件照规格（头部占 70%），1.6 为通用头肩像，2.0 为半身像")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "提取成功（图片字节）"),
		@ApiResponse(responseCode = "400", description = "未检测到人脸 / 参数越界", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> extract(
		@Parameter(description = "源图片（照片 / 证件 / 证件照均可）", required = true,
			schema = @Schema(type = "string", format = "binary"))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "输出边长，默认取 mica.ai.face.avatar.size") @RequestParam(value = "size", required = false) Integer size,
		@Parameter(description = "窗口边长/人脸框长边，默认 1.6") @RequestParam(value = "faceScale", required = false) Double faceScale,
		@Parameter(description = "裁剪中心下移量（人脸框高的倍数），默认 0") @RequestParam(value = "verticalOffset", required = false) Double verticalOffset,
		@Parameter(description = "是否按眼线去倾斜，默认 true") @RequestParam(value = "deRotate", required = false) Boolean deRotate,
		@Parameter(description = "额外旋转量（度），补足模型对小角度 roll 响应不足；正值更逆时针")
		@RequestParam(value = "rotation", required = false) Double rotation,
		@Parameter(description = "是否自动摆正 90° 级大角度朝向，默认 true")
		@RequestParam(value = "autoOrient", required = false) Boolean autoOrient,
		@Parameter(description = "越界填充色 #RRGGBB，默认 #FFFFFF") @RequestParam(value = "background", required = false) String background,
		@Parameter(description = "常规检测无果时是否启用分块检测兜底，默认 true")
		@RequestParam(value = "tileDetect", required = false) Boolean tileDetect,
		@Parameter(description = "分块边长，默认 480；越小则小脸入网放得越大")
		@RequestParam(value = "tileSize", required = false) Integer tileSize,
		@Parameter(description = "输出格式 png/jpg，默认 png") @RequestParam(value = "format", required = false) String format,
		@Parameter(description = "JPEG 质量 [1,100]，默认 95") @RequestParam(value = "quality", required = false) Integer quality)
		throws IOException {

		String fmt = format == null ? "png" : format;
		Mat image = null;
		List<AvatarResult> results = null;
		try {
			image = ImageUtils.byteArrayToMat(file.getBytes());
			AvatarOptions opts = options(size, faceScale, verticalOffset, deRotate,
				rotation, autoOrient, background, tileDetect, tileSize, 1);
			results = avatarExtractor.extractAll(image, opts);
			if (results.isEmpty()) {
				return json(error("未检测到人脸"), HttpStatus.BAD_REQUEST);
			}
			AvatarResult first = results.get(0);
			Map<String, String> headers = new LinkedHashMap<>();
			headers.put("X-Avatar-Face-Size", String.format("%.1f", first.getFaceSize()));
			headers.put("X-Avatar-Usable", String.valueOf(first.isUsable()));
			headers.put("X-Avatar-Sharpness", String.format("%.1f", first.getSharpness()));
			headers.put("X-Avatar-Orientation", String.valueOf(first.getOrientationDegrees()));
			headers.put("X-Avatar-Tiled", String.valueOf(first.isTiledDetection()));
			return image(ImageUtils.matToBytes(first.getImage(), fmt, quality(quality)),
				contentType(fmt), headers);
		} catch (MicaAiException e) {
			log.warn("头像提取失败: {}", e.getMessage());
			return failure(e);
		} finally {
			releaseAll(results);
			ImageUtils.releaseAll(image);
		}
	}

	/**
	 * 提取头像（全部人脸）：返回每张人脸的元数据与 base64 头像，按面积降序。
	 */
	@Operation(summary = "提取头像（全部人脸）",
		description = "返回检测到的每张人脸的元数据（框、像素数、清晰度、可用性、摆正角度）与 base64 头像，按面积降序")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "提取成功",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败 / 参数越界", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/faces", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Map<String, Object>> faces(
		@Parameter(description = "源图片", required = true,
			schema = @Schema(type = "string", format = "binary"))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "输出边长") @RequestParam(value = "size", required = false) Integer size,
		@Parameter(description = "窗口边长/人脸框长边，默认 1.6") @RequestParam(value = "faceScale", required = false) Double faceScale,
		@Parameter(description = "裁剪中心下移量，默认 0") @RequestParam(value = "verticalOffset", required = false) Double verticalOffset,
		@Parameter(description = "是否按眼线去倾斜，默认 true") @RequestParam(value = "deRotate", required = false) Boolean deRotate,
		@Parameter(description = "额外旋转量（度），补足模型对小角度 roll 响应不足；正值更逆时针")
		@RequestParam(value = "rotation", required = false) Double rotation,
		@Parameter(description = "是否自动摆正 90° 级大角度朝向，默认 true")
		@RequestParam(value = "autoOrient", required = false) Boolean autoOrient,
		@Parameter(description = "越界填充色 #RRGGBB") @RequestParam(value = "background", required = false) String background,
		@Parameter(description = "常规检测无果时是否启用分块检测兜底") @RequestParam(value = "tileDetect", required = false) Boolean tileDetect,
		@Parameter(description = "分块边长，默认 480") @RequestParam(value = "tileSize", required = false) Integer tileSize,
		@Parameter(description = "输出格式 png/jpg") @RequestParam(value = "format", required = false) String format,
		@Parameter(description = "JPEG 质量 [1,100]") @RequestParam(value = "quality", required = false) Integer quality,
		@Parameter(description = "最多返回人脸数，0 为不限") @RequestParam(value = "maxFaces", required = false) Integer maxFaces)
		throws IOException {

		String fmt = format == null ? "png" : format;
		Mat image = null;
		List<AvatarResult> results = null;
		try {
			image = ImageUtils.byteArrayToMat(file.getBytes());
			AvatarOptions opts = options(size, faceScale, verticalOffset, deRotate,
				rotation, autoOrient, background, tileDetect, tileSize, maxFaces);
			results = avatarExtractor.extractAll(image, opts);

			List<Map<String, Object>> faceList = new ArrayList<>(results.size());
			for (AvatarResult r : results) {
				FaceBox b = r.getBox();
				Map<String, Object> fm = new LinkedHashMap<>();
				fm.put("index", r.getIndex());
				fm.put("score", b.getScore());
				fm.put("bbox", new float[]{b.getX1(), b.getY1(), b.getX2(), b.getY2()});
				fm.put("landmarks", b.getLandmarks());
				fm.put("windowQuad", r.getWindowQuad());
				fm.put("faceSize", r.getFaceSize());
				fm.put("usable", r.isUsable());
				fm.put("unusableReason", r.getUnusableReason());
				fm.put("sharpness", r.getSharpness());
				fm.put("orientationDegrees", r.getOrientationDegrees());
				fm.put("tiledDetection", r.isTiledDetection());
				fm.put("format", fmt);
				fm.put("imageBase64", Base64.getEncoder()
					.encodeToString(ImageUtils.matToBytes(r.getImage(), fmt, quality(quality))));
				faceList.add(fm);
			}
			Map<String, Object> ok = new LinkedHashMap<>();
			ok.put("count", faceList.size());
			ok.put("faces", faceList);
			return ResponseEntity.ok(ok);
		} catch (MicaAiException e) {
			log.warn("头像提取失败: {}", e.getMessage());
			Map<String, Object> body = error(e.getMessage());
			return ResponseEntity.badRequest().body(body);
		} finally {
			releaseAll(results);
			ImageUtils.releaseAll(image);
		}
	}

	/**
	 * 调试图：在原图上画出人脸框（红）与头像裁剪窗口（绿）。
	 */
	@Operation(summary = "头像取景调试图",
		description = "在原图上画出人脸框（红）、头像裁剪窗口（绿）与人脸像素数，用于人工核对取景；"
			+ "坐标始终位于原图，即使做了自动定向")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "标注成功（图片字节）"),
		@ApiResponse(responseCode = "400", description = "参数越界", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/annotate", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> annotate(
		@Parameter(description = "源图片", required = true,
			schema = @Schema(type = "string", format = "binary"))
		@RequestPart("file") MultipartFile file,
		@Parameter(description = "窗口边长/人脸框长边，默认 1.6") @RequestParam(value = "faceScale", required = false) Double faceScale,
		@Parameter(description = "裁剪中心下移量，默认 0") @RequestParam(value = "verticalOffset", required = false) Double verticalOffset,
		@Parameter(description = "是否按眼线去倾斜，默认 true") @RequestParam(value = "deRotate", required = false) Boolean deRotate,
		@Parameter(description = "额外旋转量（度），补足模型对小角度 roll 响应不足；正值更逆时针")
		@RequestParam(value = "rotation", required = false) Double rotation,
		@Parameter(description = "是否自动摆正 90° 级大角度朝向，默认 true")
		@RequestParam(value = "autoOrient", required = false) Boolean autoOrient,
		@Parameter(description = "常规检测无果时是否启用分块检测兜底") @RequestParam(value = "tileDetect", required = false) Boolean tileDetect,
		@Parameter(description = "分块边长，默认 480") @RequestParam(value = "tileSize", required = false) Integer tileSize,
		@Parameter(description = "输出格式 png/jpg，默认 png") @RequestParam(value = "format", required = false) String format,
		@Parameter(description = "JPEG 质量 [1,100]，默认 95") @RequestParam(value = "quality", required = false) Integer quality)
		throws IOException {

		String fmt = format == null ? "png" : format;
		Mat image = null;
		Mat canvas = null;
		List<AvatarResult> results = null;
		try {
			image = ImageUtils.byteArrayToMat(file.getBytes());
			AvatarOptions opts = options(null, faceScale, verticalOffset, deRotate,
				rotation, autoOrient, null, tileDetect, tileSize, 0);
			results = avatarExtractor.extractAll(image, opts);
			canvas = AvatarExtractor.drawDebug(image, results, opts);
			return image(ImageUtils.matToBytes(canvas, fmt, quality(quality)),
				contentType(fmt), null);
		} catch (MicaAiException e) {
			log.warn("头像标注失败: {}", e.getMessage());
			return failure(e);
		} finally {
			releaseAll(results);
			ImageUtils.releaseAll(canvas, image);
		}
	}

	/**
	 * 以配置的默认为基准，用非空的查询参数覆盖。
	 */
	private AvatarOptions options(Integer size, Double faceScale, Double verticalOffset,
								  Boolean deRotate, Double rotation, Boolean autoOrient,
								  String background, Boolean tileDetect, Integer tileSize,
								  Integer maxFaces) {
		AvatarOptions.AvatarOptionsBuilder builder = AvatarOptions.defaults().toBuilder();
		if (size != null) {
			builder.size(size);
		}
		if (faceScale != null) {
			builder.faceScale(faceScale);
		}
		if (verticalOffset != null) {
			builder.verticalOffset(verticalOffset);
		}
		if (deRotate != null) {
			builder.deRotate(deRotate);
		}
		if (rotation != null) {
			builder.rotationDegrees(rotation);
		}
		if (autoOrient != null) {
			builder.autoOrient(autoOrient);
		}
		if (background != null && !background.isEmpty()) {
			builder.background(background);
		}
		if (tileDetect != null) {
			builder.tileDetect(tileDetect);
		}
		if (tileSize != null) {
			builder.tileSize(tileSize);
		}
		if (maxFaces != null) {
			builder.maxFaces(maxFaces);
		}
		return builder.build();
	}

	private static int quality(Integer quality) {
		return quality == null ? 95 : quality;
	}

	private static MediaType contentType(String format) {
		return "jpg".equalsIgnoreCase(format) || "jpeg".equalsIgnoreCase(format)
			? MediaType.IMAGE_JPEG : MediaType.IMAGE_PNG;
	}

	private static ResponseEntity<Object> image(byte[] bytes, MediaType type,
												Map<String, String> headers) {
		ResponseEntity.BodyBuilder builder = ResponseEntity.ok().contentType(type);
		if (headers != null) {
			headers.forEach(builder::header);
		}
		return builder.body(bytes);
	}

	private static ResponseEntity<Object> json(Object body, HttpStatus status) {
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(body);
	}

	/**
	 * {@link MicaAiException} 统一按 400 返回：本接口的异常基本都是「未检测到人脸」
	 * 或参数越界，属于调用方问题。
	 */
	private static ResponseEntity<Object> failure(MicaAiException e) {
		Map<String, Object> body = error(e.getMessage());
		return json(body, HttpStatus.BAD_REQUEST);
	}

	private static void releaseAll(List<AvatarResult> results) {
		if (results != null) {
			results.forEach(AvatarResult::release);
		}
	}

	private static Map<String, Object> error(String msg) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return m;
	}
}
