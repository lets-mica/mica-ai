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
import net.dreamlu.mica.ai.detect.core.SlicedDetectEngine;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import net.dreamlu.mica.ai.detect.sliced.Slice;
import net.dreamlu.mica.ai.detect.sliced.SliceGrid;
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
 * 切片推理 REST 端点（提升小目标召回）。
 *
 * <p>切片边长恒等于模型输入边长，重叠率、切片数上限、合并策略取 starter 中
 * {@code mica.ai.detect.sliced.*} 配置。
 */
@Slf4j
@Tag(name = "Detect 切片推理", description = "大图切片推理 · 小目标召回增强 · 整图/切片对比")
@RestController
@RequestMapping("/detect")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.detect.sliced", name = "enabled",
	havingValue = "true", matchIfMissing = true)
public class SlicedDetectController {

	private final DetectEngine detectEngine;

	private final SlicedDetectEngine slicedDetectEngine;

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

	private static List<Map<String, Object>> boxViews(List<DetectBox> boxes, String[] labels) {
		List<Map<String, Object>> views = new ArrayList<>();
		for (DetectBox box : boxes) {
			Map<String, Object> m = new LinkedHashMap<>();
			m.put("label", box.label(labels));
			m.put("classId", box.getClassId());
			m.put("score", box.getScore());
			m.put("x1", box.getX1());
			m.put("y1", box.getY1());
			m.put("x2", box.getX2());
			m.put("y2", box.getY2());
			views.add(m);
		}
		return views;
	}

	/**
	 * 按 IoU 找出只在其中一侧出现的框：整图漏掉、切片抓到的目标就是切片的价值所在。
	 */
	private static List<DetectBox> unmatched(List<DetectBox> source, List<DetectBox> reference,
										   float iouThreshold) {
		List<DetectBox> result = new ArrayList<>();
		for (DetectBox box : source) {
			boolean matched = false;
			for (DetectBox other : reference) {
				if (box.iou(other) > iouThreshold) {
					matched = true;
					break;
				}
			}
			if (!matched) {
				result.add(box);
			}
		}
		return result;
	}

	@Operation(summary = "切片目标检测",
		description = "大图切成多片分别推理后合并，返回原图坐标系的检测框；比整图推理更擅长小目标")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "检测成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败或切片数超限", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/sliced", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> sliced(
		@Parameter(description = "待检测图片（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		Mat image = null;
		try {
			image = DetectImageUtils.byteArrayToMat(file.getBytes());
			long start = System.nanoTime();
			DetectResult result = slicedDetectEngine.detect(image);
			long elapsedMs = (System.nanoTime() - start) / 1_000_000L;

			Map<String, Object> body = new LinkedHashMap<>();
			body.put("width", result.getWidth());
			body.put("height", result.getHeight());
			body.put("count", result.size());
			body.put("sliceCount", slicedDetectEngine.sliceCount(result.getWidth(), result.getHeight()));
			body.put("elapsedMs", elapsedMs);
			body.put("boxes", boxViews(result.getBoxes(), detectEngine.getLabels()));
			return json(body);
		} catch (MicaAiException e) {
			log.warn("切片目标检测失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		} finally {
			if (image != null) {
				image.release();
			}
		}
	}

	@Operation(summary = "整图 vs 切片对比",
		description = "同一张图同时跑整图推理与切片推理，返回两侧框数与耗时、"
			+ "只在切片出现的框（整图漏掉、切片抓到）以及算力代价倍数")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "对比成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败或切片数超限", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/sliced-compare", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> slicedCompare(
		@Parameter(description = "待检测图片（jpg / png 等）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		Mat image = null;
		try {
			image = DetectImageUtils.byteArrayToMat(file.getBytes());

			long wholeStart = System.nanoTime();
			DetectResult whole = detectEngine.detect(image);
			long wholeMs = (System.nanoTime() - wholeStart) / 1_000_000L;

			long slicedStart = System.nanoTime();
			DetectResult sliced = slicedDetectEngine.detect(image);
			long slicedMs = (System.nanoTime() - slicedStart) / 1_000_000L;

			float iouThreshold = detectEngine.getConfig().getNmsThreshold();
			List<DetectBox> onlyInSliced = unmatched(sliced.getBoxes(), whole.getBoxes(), iouThreshold);
			List<DetectBox> onlyInWhole = unmatched(whole.getBoxes(), sliced.getBoxes(), iouThreshold);
			String[] labels = detectEngine.getLabels();

			Map<String, Object> wholeView = new LinkedHashMap<>();
			wholeView.put("count", whole.size());
			wholeView.put("elapsedMs", wholeMs);
			wholeView.put("boxes", boxViews(whole.getBoxes(), labels));

			Map<String, Object> slicedView = new LinkedHashMap<>();
			slicedView.put("count", sliced.size());
			slicedView.put("elapsedMs", slicedMs);
			slicedView.put("sliceCount", slicedDetectEngine.sliceCount(sliced.getWidth(), sliced.getHeight()));
			slicedView.put("boxes", boxViews(sliced.getBoxes(), labels));

			Map<String, Object> body = new LinkedHashMap<>();
			body.put("width", whole.getWidth());
			body.put("height", whole.getHeight());
			body.put("whole", wholeView);
			body.put("sliced", slicedView);
			body.put("onlyInSliced", boxViews(onlyInSliced, labels));
			body.put("onlyInWhole", boxViews(onlyInWhole, labels));
			body.put("extraCostRatio", wholeMs <= 0 ? 0d : Math.round((double) slicedMs / wholeMs * 100d) / 100d);
			return json(body);
		} catch (MicaAiException e) {
			log.warn("整图切片对比失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		} finally {
			if (image != null) {
				image.release();
			}
		}
	}

	@Operation(summary = "切片检测调试图",
		description = "在原图上画出切片推理得到的检测框与类别名，输出 PNG")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "标注成功（PNG 字节）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/sliced-annotate", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> slicedAnnotate(
		@Parameter(description = "待检测图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		Mat canvas = null;
		MatOfByte buf = null;
		try {
			canvas = DetectImageUtils.byteArrayToMat(file.getBytes());
			DetectResult result = slicedDetectEngine.detect(canvas);
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
			log.warn("切片检测标注失败: {}", e.getMessage());
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

	@Operation(summary = "切片网格可视化",
		description = "在原图上叠加切片网格与片序号，直观展示切片边长、重叠与末列回退，输出 PNG")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "绘制成功（PNG 字节）"),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/sliced-grid", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> slicedGrid(
		@Parameter(description = "待切分的图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		Mat canvas = null;
		MatOfByte buf = null;
		try {
			canvas = DetectImageUtils.byteArrayToMat(file.getBytes());
			int width = canvas.cols();
			int height = canvas.rows();
			int size = slicedDetectEngine.getConfig().resolveSliceSize();
			int step = slicedDetectEngine.getConfig().resolveStep();
			SliceGrid grid = SliceGrid.of(width, height, size, step);

			Scalar lineColor = new Scalar(255, 128, 0);
			for (Slice slice : grid.getSlices()) {
				Imgproc.rectangle(canvas,
					new Point(slice.getX(), slice.getY()),
					new Point(Math.min(slice.getX2(), width), Math.min(slice.getY2(), height)),
					lineColor, 1);
			}
			int index = 0;
			for (Slice slice : grid.getSlices()) {
				Imgproc.putText(canvas, String.valueOf(index++),
					new Point(slice.getX() + 4, slice.getY() + 16),
					Imgproc.FONT_HERSHEY_SIMPLEX, 0.45, lineColor, 1);
			}

			buf = new MatOfByte();
			Imgcodecs.imencode(".png", canvas, buf);
			return image(buf.toArray());
		} catch (IllegalArgumentException e) {
			log.warn("切片网格绘制失败: {}", e.getMessage());
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
