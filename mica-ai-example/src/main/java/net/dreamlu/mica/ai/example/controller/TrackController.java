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
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.detect.core.DetectEngine;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import net.dreamlu.mica.ai.track.core.ByteTracker;
import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
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
 * ByteTrack 多目标跟踪 REST 端点（纯算法零模型）。
 * <p>
 * 跟踪器是<b>有状态</b>的：同一个实例要按顺序喂入同一路视频的连续帧，
 * 输出的 track id 才会稳定。示例里注入的是 starter 提供的单例 {@link ByteTracker}，
 * 真实业务中请「一路流一个实例」（按流 key 缓存），并自行加锁避免并发喂帧。
 * </p>
 */
@Slf4j
@Tag(name = "Track 多目标跟踪", description = "ByteTrack · Kalman 滤波 + 匈牙利匹配，稳定 track id")
@RestController
@RequestMapping("/track")
@RequiredArgsConstructor
@ConditionalOnProperty(prefix = "mica.ai.track", name = "enabled", havingValue = "true", matchIfMissing = true)
public class TrackController {

	private final ByteTracker tracker;
	/**
	 * 检测能力可关闭，故用 ObjectProvider 注入；未启用时仅 {@code /track/update} 可用。
	 */
	private final ObjectProvider<DetectEngine> detectEngineProvider;

	/**
	 * 单帧检测框请求体。
	 */
	@Data
	public static class BoxRequest {

		/**
		 * 左上角 x
		 */
		private float x1;
		/**
		 * 左上角 y
		 */
		private float y1;
		/**
		 * 右下角 x
		 */
		private float x2;
		/**
		 * 右下角 y
		 */
		private float y2;
		/**
		 * 置信度，缺省 0.5
		 */
		private float score = 0.5f;
		/**
		 * 类别 id，缺省 0
		 */
		private int classId = 0;

		BBox toBBox() {
			return new BBox(x1, y1, x2, y2, score, classId);
		}
	}

	private static ResponseEntity<Object> json(Map<String, Object> body) {
		return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

	private Map<String, Object> trackView(Track track, int confirmAge) {
		BBox box = track.getCurrentBBox();
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("id", track.getId());
		m.put("x1", box.getX1());
		m.put("y1", box.getY1());
		m.put("x2", box.getX2());
		m.put("y2", box.getY2());
		m.put("score", track.getScore());
		m.put("classId", track.getClassId());
		m.put("age", track.getAge());
		m.put("lostFrames", track.getLostFrames());
		m.put("mature", track.isMature(confirmAge));
		return m;
	}

	private Map<String, Object> trackBody(List<Track> tracks) {
		int confirmAge = tracker.getConfig().getConfirmAge();
		List<Map<String, Object>> items = new ArrayList<>(tracks.size());
		for (Track track : tracks) {
			items.add(trackView(track, confirmAge));
		}
		Map<String, Object> body = new LinkedHashMap<>();
		body.put("count", items.size());
		body.put("tracks", items);
		return body;
	}

	/**
	 * 喂一帧检测框，返回当前活跃轨迹。
	 */
	@Operation(summary = "跟踪一帧检测框",
		description = "请求体是一帧的检测框数组（x1/y1/x2/y2/score/classId），"
			+ "返回当前活跃轨迹（id / 框 / score / age / mature）。"
			+ "跟踪器有状态，需按顺序连续喂同一路视频的帧")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "跟踪成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "请求体非法", content = @Content)
	})
	@PostMapping(value = "/update", consumes = MediaType.APPLICATION_JSON_VALUE)
	public ResponseEntity<Object> update(
		@Parameter(description = "一帧的检测框数组", required = true)
		@RequestBody List<BoxRequest> boxes) {
		try {
			List<BBox> dets = new ArrayList<>();
			if (boxes != null) {
				for (BoxRequest box : boxes) {
					if (box != null) {
						dets.add(box.toBBox());
					}
				}
			}
			return json(trackBody(tracker.update(dets)));
		} catch (MicaAiException e) {
			log.warn("跟踪失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	/**
	 * 检测 + 跟踪：上传一帧图片，自动检测目标并喂给跟踪器，返回带 track id 的框。
	 */
	@Operation(summary = "检测 + 跟踪（一键）",
		description = "上传一帧图片，内部先做通用目标检测，再把检测框喂给 ByteTrack，"
			+ "返回每个检测框对应的 track id（跨帧稳定）")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "跟踪成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "图片读取失败", content = @Content),
		@ApiResponse(responseCode = "500", description = "推理异常", content = @Content)
	})
	@PostMapping(value = "/detect-and-track", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
	public ResponseEntity<Object> detectAndTrack(
		@Parameter(description = "待处理的一帧图片", required = true,
			content = @Content(mediaType = MediaType.MULTIPART_FORM_DATA_VALUE,
				schema = @Schema(type = "string", format = "binary")))
		@RequestPart("file") MultipartFile file) throws IOException {
		DetectEngine detectEngine = detectEngineProvider.getIfAvailable();
		if (detectEngine == null) {
			return error("目标检测能力未启用（mica.ai.detect.enabled=false），无法执行检测 + 跟踪",
				HttpStatus.SERVICE_UNAVAILABLE);
		}
		try {
			DetectResult result = detectEngine.detectBytes(file.getBytes());
			List<BBox> dets = new ArrayList<>();
			for (DetectBox box : result.getBoxes()) {
				dets.add(new BBox(box.getX1(), box.getY1(), box.getX2(), box.getY2(),
					box.getScore(), box.getClassId()));
			}
			List<Track> tracks = tracker.update(dets);

			Map<String, Object> body = new LinkedHashMap<>();
			body.put("width", result.getWidth());
			body.put("height", result.getHeight());
			body.put("detections", result.size());
			List<Map<String, Object>> items = new ArrayList<>();
			int confirmAge = tracker.getConfig().getConfirmAge();
			for (Track track : tracks) {
				BBox box = track.getCurrentBBox();
				Map<String, Object> m = trackView(track, confirmAge);
				m.put("label", labelOf(detectEngine, track.getClassId()));
				m.put("trackId", track.getId());
				m.put("trackBox", new float[]{box.getX1(), box.getY1(), box.getX2(), box.getY2()});
				items.add(m);
			}
			body.put("tracks", items);
			return json(body);
		} catch (MicaAiException e) {
			log.warn("检测 + 跟踪失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	private String labelOf(DetectEngine detectEngine, int classId) {
		String[] labels = detectEngine.getLabels();
		if (labels == null || classId < 0 || classId >= labels.length) {
			return String.valueOf(classId);
		}
		return labels[classId];
	}
}
