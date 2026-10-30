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
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.detect.core.DetectEngine;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import net.dreamlu.mica.ai.detect.util.DetectImageUtils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 通用目标检测 REST 端点（NanoDet-Plus / YOLOX-s，COCO 80 类）。
 * <p>
 * 模型族、输入边长、置信度阈值等取 starter 中 {@code mica.ai.detect.*} 配置；
 * 换模型族时 {@code family} 与 {@code input-size} 必须成对修改。
 * </p>
 */
@Slf4j
@Tag(name = "Detect 通用目标检测", description = "NanoDet-Plus / YOLOX-s · COCO 80 类目标检测")
@RestController
@RequestMapping("/detect")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.detect", name = "enabled", havingValue = "true", matchIfMissing = true)
public class DetectController {

	private final DetectEngine detectEngine;

	private static ResponseEntity<Object> json(Map<String, Object> body) {
		return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> image(byte[] bytes) {
		return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).body(bytes);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

	/**
	 * 检测目标：返回每个框的类别名、置信度与原图坐标。
	 */
	@Operation(summary = "通用目标检测",
		description = "返回原图坐标系下的检测框（label / classId / score / x1,y1,x2,y2）与图片尺寸")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "检测成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/detect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> detect(
		@Parameter(description = "待检测图片（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		try {
			DetectResult result = detectEngine.detectBytes(file.getBytes());
			Map<String, Object> body = new LinkedHashMap<>();
			body.put("width", result.getWidth());
			body.put("height", result.getHeight());
			body.put("count", result.size());
			List<Map<String, Object>> boxes = new ArrayList<>();
			for (DetectBox box : result.getBoxes()) {
				Map<String, Object> m = new LinkedHashMap<>();
				m.put("label", box.label(detectEngine.getLabels()));
				m.put("classId", box.getClassId());
				m.put("score", box.getScore());
				m.put("x1", box.getX1());
				m.put("y1", box.getY1());
				m.put("x2", box.getX2());
				m.put("y2", box.getY2());
				boxes.add(m);
			}
			body.put("boxes", boxes);
			return json(body);
		} catch (MicaAiException e) {
			log.warn("目标检测失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	/**
	 * 检测并标注：在原图上画出检测框与类别名，输出 PNG。
	 */
	@Operation(summary = "目标检测调试图",
		description = "在原图上画出检测框（绿）与类别名，输出 PNG，便于人工核对框位")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "标注成功（PNG 字节）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/annotate", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> annotate(
		@Parameter(description = "待检测图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		Mat canvas = null;
		MatOfByte buf = null;
		try {
			canvas = DetectImageUtils.byteArrayToMat(file.getBytes());
			DetectResult result = detectEngine.detect(canvas);
			String[] labels = detectEngine.getLabels();
			for (DetectBox box : result.getBoxes()) {
				Point p1 = new Point(Math.round(box.getX1()), Math.round(box.getY1()));
				Point p2 = new Point(Math.round(box.getX2()), Math.round(box.getY2()));
				Imgproc.rectangle(canvas, p1, p2, new Scalar(0, 255, 0), 2);
				Imgproc.putText(canvas, box.label(labels), p1,
					Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(0, 255, 0), 2);
			}
			buf = new MatOfByte();
			Imgcodecs.imencode(".png", canvas, buf);
			return image(buf.toArray());
		} catch (MicaAiException e) {
			log.warn("目标检测标注失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		} finally {
			if (buf != null) {
				buf.release();
			}
			if (canvas != null) {
				canvas.release();
			}
		}
	}
}
