/*
 * Copyright (c) 2019-2029, DreamLu 卢春梦 (596392912@qq.com & dreamlu.net).
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

import com.fasterxml.jackson.databind.ObjectMapper;
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
import net.dreamlu.mica.ai.example.pipeline.inspection.InspectionOptions;
import net.dreamlu.mica.ai.example.pipeline.inspection.InspectionPipeline;
import net.dreamlu.mica.ai.example.pipeline.inspection.InspectionResult;
import net.dreamlu.mica.ai.track.model.TrackEvent;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 小球机巡检场景 REST 端点。
 *
 * <p>把「多帧监控画面 + 每帧相机运动」跑完 <b>切片检测 → 运动补偿跟踪 → 行为事件</b> 这条链：
 * 切片负责「看得见」（远处小目标），补偿负责「跟得住」（相机一动 ID 不断裂），
 * 事件规则负责「判得准」（越线 / ROI 进出 / 滞留 / 折返）。
 *
 * @author DreamLu
 */
@Slf4j
@Tag(name = "Scenario 小球机巡检", description = "切片检测 → 相机运动补偿跟踪 → 越线 / ROI / 滞留 / 折返")
@RestController
@RequestMapping("/scenario/inspection")
@RequiredArgsConstructor
@ConditionalOnBean({DetectEngine.class, SlicedDetectEngine.class})
public class InspectionScenarioController {

	private static final ObjectMapper MAPPER = new ObjectMapper();

	private final DetectEngine detectEngine;

	private final SlicedDetectEngine slicedDetectEngine;

	@Operation(summary = "巡检分析",
		description = "上传多帧监控画面（按时间顺序），可选带上每帧相机运动增量，"
			+ "返回逐帧轨迹（图像坐标 + 参考帧坐标）、事件流与汇总")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "分析成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "参数非法或图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/analyze", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> analyze(
		@Parameter(description = "各帧图片（同一 part 名，按时间顺序）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("files") List<MultipartFile> files,
		@Parameter(description = "规则与开关 JSON，可省略（用默认值）")
		@RequestParam(value = "options", required = false) String optionsJson) {
		if (files == null || files.isEmpty()) {
			return error("files 不能为空，至少需要一帧", HttpStatus.BAD_REQUEST);
		}
		try {
			InspectionOptions options = parseOptions(optionsJson);
			InspectionResult result = pipeline().process(readAll(files), options);
			return json(report(result));
		} catch (MicaAiException e) {
			log.warn("巡检分析失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		} catch (IllegalArgumentException e) {
			log.warn("巡检分析参数非法: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		} catch (IOException e) {
			log.warn("巡检分析读取图片失败: {}", e.getMessage());
			return error("图片读取失败: " + e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	@Operation(summary = "巡检标注图",
		description = "参数同上，返回最后一帧的标注 PNG（轨迹框 + id + 轨迹点连线）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "绘制成功（PNG 字节）"),
		@ApiResponse(responseCode = "400", description = "参数非法或图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/annotate", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<byte[]> annotate(
		@Parameter(description = "各帧图片（同一 part 名，按时间顺序）", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("files") List<MultipartFile> files,
		@Parameter(description = "规则与开关 JSON，可省略（用默认值）")
		@RequestParam(value = "options", required = false) String optionsJson) {
		if (files == null || files.isEmpty()) {
			throw new IllegalArgumentException("files 不能为空，至少需要一帧");
		}
		try {
			InspectionOptions options = parseOptions(optionsJson);
			InspectionResult result = pipeline().process(readAll(files), options);
			byte[] png = result.getAnnotatedPng();
			if (png == null) {
				throw new IllegalArgumentException("没有可标注的帧");
			}
			return ResponseEntity.ok().contentType(MediaType.IMAGE_PNG).body(png);
		} catch (IOException e) {
			throw new IllegalArgumentException("图片读取失败: " + e.getMessage(), e);
		}
	}

	private InspectionPipeline pipeline() {
		return new InspectionPipeline(detectEngine, slicedDetectEngine, detectEngine.getLabels());
	}

	private static List<byte[]> readAll(List<MultipartFile> files) throws IOException {
		List<byte[]> result = new ArrayList<>();
		for (MultipartFile file : files) {
			if (file != null && !file.isEmpty()) {
				result.add(file.getBytes());
			}
		}
		return result;
	}

	private static InspectionOptions parseOptions(String optionsJson) {
		if (optionsJson == null || optionsJson.trim().isEmpty()) {
			return new InspectionOptions();
		}
		try {
			return MAPPER.readValue(optionsJson, InspectionOptions.class);
		} catch (IOException e) {
			throw new IllegalArgumentException("options 不是合法 JSON: " + e.getMessage(), e);
		}
	}

	/**
	 * 把巡检结果摊平成 JSON 报告。
	 */
	private static Map<String, Object> report(InspectionResult result) {
		List<Map<String, Object>> frames = new ArrayList<>();
		for (InspectionResult.FrameReport frame : result.getFrames()) {
			List<Map<String, Object>> tracks = new ArrayList<>();
			for (InspectionResult.TrackView track : frame.getTracks()) {
				Map<String, Object> item = new LinkedHashMap<>();
				item.put("id", track.getId());
				item.put("label", track.getLabel());
				item.put("classId", track.getClassId());
				item.put("score", track.getScore());
				item.put("cx", track.getCx());
				item.put("cy", track.getCy());
				item.put("refCx", track.getRefCx());
				item.put("refCy", track.getRefCy());
				item.put("speedPxPerSec", track.getSpeedPxPerSec());
				item.put("direction", track.getDirection());
				item.put("age", track.getAge());
				tracks.add(item);
			}
			Map<String, Object> item = new LinkedHashMap<>();
			item.put("frameIndex", frame.getFrameIndex());
			item.put("timestampMs", frame.getTimestampMs());
			item.put("sliceCount", frame.getSliceCount());
			item.put("detectCount", frame.getDetectCount());
			item.put("tracks", tracks);
			item.put("events", eventViews(frame.getEvents()));
			frames.add(item);
		}

		Map<String, Object> body = new LinkedHashMap<>();
		body.put("frameCount", result.getFrames().size());
		body.put("frames", frames);
		body.put("events", eventViews(result.getEvents()));
		body.put("summary", result.getSummary());
		return body;
	}

	private static List<Map<String, Object>> eventViews(List<TrackEvent> events) {
		List<Map<String, Object>> views = new ArrayList<>();
		for (TrackEvent event : events) {
			Map<String, Object> item = new LinkedHashMap<>();
			item.put("type", event.getType().name());
			item.put("trackId", event.getTrackId());
			item.put("classId", event.getClassId());
			item.put("frameIndex", event.getFrameIndex());
			item.put("timestampMs", event.getTimestampMs());
			item.put("ruleName", event.getRuleName());
			item.put("direction", event.getDirection());
			item.put("detail", event.getDetail());
			views.add(item);
		}
		return views;
	}

	private static ResponseEntity<Object> json(Map<String, Object> body) {
		return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

}
