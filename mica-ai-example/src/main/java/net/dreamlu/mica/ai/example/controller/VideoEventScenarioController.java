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
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.track.config.ByteTrackConfig;
import net.dreamlu.mica.ai.track.event.AnchorPolicy;
import net.dreamlu.mica.ai.track.event.DirectionRule;
import net.dreamlu.mica.ai.track.event.DwellRule;
import net.dreamlu.mica.ai.track.event.FrameResult;
import net.dreamlu.mica.ai.track.event.LineCrossingRule;
import net.dreamlu.mica.ai.track.event.Roi;
import net.dreamlu.mica.ai.track.event.RoiCounterRule;
import net.dreamlu.mica.ai.track.event.TrackEventEngine;
import net.dreamlu.mica.ai.track.model.BBox;
import net.dreamlu.mica.ai.track.model.Track;
import net.dreamlu.mica.ai.track.model.TrackEvent;
import net.dreamlu.mica.ai.track.model.TrackEventType;
import net.dreamlu.mica.ai.track.model.TrackPoint;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 视频图像 · 逐帧事件分析场景。
 *
 * <p>把「连续帧的检测框 + 每帧采集时间戳」喂给 {@link TrackEventEngine}，输出稳定轨迹与
 * 越线 / ROI 进出 / 滞留 / 折返事件。本示例刻意**不解码视频**：真实链路里
 * 帧由 mica-video（JavaCV / FFmpeg）提供，检测器把 {@code Mat} 转成 {@code List<BBox>} 后调用这里。
 *
 * <p>每次请求新建引擎（跟踪器与规则都是有状态的），一路流一个实例，请求之间不串状态。
 */
@Slf4j
@Tag(name = "视频事件分析", description = "逐帧检测框 → ByteTrack 轨迹 → 越线 / ROI / 滞留 / 折返事件")
@RestController
@RequestMapping("/scenario/video-events")
@ConditionalOnProperty(prefix = "mica.ai.track", name = "enabled", havingValue = "true", matchIfMissing = true)
public class VideoEventScenarioController {

	@Data
	public static class AnalyzeRequest {
		private List<FrameRequest> frames;
		private LineRequest line = new LineRequest();
		private RoiRequest roi = new RoiRequest();
		private long dwellThresholdMs = 1000L;
	}

	@Data
	public static class FrameRequest {
		private long timestampMs;
		private List<BoxRequest> boxes;
	}

	@Data
	public static class BoxRequest {
		private float x1;
		private float y1;
		private float x2;
		private float y2;
		private float score = 0.9f;
		private int classId = 0;
	}

	@Data
	public static class LineRequest {
		private String name = "gate";
		private float x1 = 100f;
		private float y1 = 0f;
		private float x2 = 100f;
		private float y2 = 400f;
	}

	@Data
	public static class RoiRequest {
		private String name = "zone";
		private float x1 = 0f;
		private float y1 = 0f;
		private float x2 = 200f;
		private float y2 = 200f;
		private boolean bottomCenterAnchor = true;
	}

	private static ResponseEntity<Object> json(Map<String, Object> body) {
		return ResponseEntity.ok().contentType(MediaType.APPLICATION_JSON).body(body);
	}

	private static ResponseEntity<Object> error(String msg, HttpStatus status) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("error", msg);
		return ResponseEntity.status(status).contentType(MediaType.APPLICATION_JSON).body(m);
	}

	private static ByteTrackConfig trackConfig() {
		return ByteTrackConfig.builder()
			.highThreshold(0.5f)
			.lowThreshold(0.01f)
			.matchThresh(0.3f)
			.matchLowThresh(0.1f)
			.confirmAge(2)
			.trackBufferedThresh(30)
			.historySize(32)
			.build();
	}

	private static List<BBox> toBoxes(List<BoxRequest> boxes) {
		List<BBox> result = new ArrayList<>();
		if (boxes == null) {
			return result;
		}
		for (BoxRequest box : boxes) {
			if (box == null) {
				continue;
			}
			result.add(new BBox(box.getX1(), box.getY1(), box.getX2(), box.getY2(),
				box.getScore(), box.getClassId()));
		}
		return result;
	}

	private static Map<String, Object> eventView(TrackEvent event) {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("type", event.getType().name());
		m.put("trackId", event.getTrackId());
		m.put("classId", event.getClassId());
		m.put("frameIndex", event.getFrameIndex());
		m.put("timestampMs", event.getTimestampMs());
		m.put("ruleName", event.getRuleName());
		m.put("direction", event.getDirection());
		m.put("detail", event.getDetail());
		m.put("box", new float[]{event.getBox().getX1(), event.getBox().getY1(),
			event.getBox().getX2(), event.getBox().getY2()});
		return m;
	}

	private static Map<String, Object> trackView(Track track) {
		TrackPoint point = track.getLastPoint();
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("id", track.getId());
		m.put("classId", track.getClassId());
		m.put("cx", point == null ? 0f : point.getCx());
		m.put("cy", point == null ? 0f : point.getCy());
		m.put("direction", track.directionLabel());
		m.put("speedPxPerSec", track.speedPerSecond());
		m.put("age", track.getAge());
		m.put("lostFrames", track.getLostFrames());
		return m;
	}

	@Operation(summary = "逐帧事件分析",
		description = "请求体是连续帧的检测框数组（每帧带 timestampMs），按顺序喂给 ByteTrack，"
			+ "返回每帧轨迹与越线 / ROI 进出 / 滞留 / 折返事件；帧来源与解码由调用方负责")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "分析成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "请求体非法或时间戳倒退", content = @Content)
	})
	@PostMapping(value = "/analyze", consumes = MediaType.APPLICATION_JSON_VALUE)
	public ResponseEntity<Object> analyze(@RequestBody AnalyzeRequest request) {
		if (request == null || request.getFrames() == null || request.getFrames().isEmpty()) {
			return error("frames 不能为空，至少需要一帧", HttpStatus.BAD_REQUEST);
		}
		try {
			LineRequest line = request.getLine() == null ? new LineRequest() : request.getLine();
			RoiRequest roi = request.getRoi() == null ? new RoiRequest() : request.getRoi();
			Roi zone = Roi.rectangle(roi.getX1(), roi.getY1(), roi.getX2(), roi.getY2());
			AnchorPolicy anchor = roi.isBottomCenterAnchor()
				? AnchorPolicy.BOTTOM_CENTER : AnchorPolicy.CENTER;
			RoiCounterRule counter = new RoiCounterRule(roi.getName(), zone, anchor, false);
			LineCrossingRule crossing = new LineCrossingRule(line.getName(),
				line.getX1(), line.getY1(), line.getX2(), line.getY2(),
				LineCrossingRule.Direction.BOTH, false, true);
			DwellRule dwell = new DwellRule(roi.getName(), zone,
				request.getDwellThresholdMs(), anchor, false);
			DirectionRule direction = new DirectionRule("u-turn", 120d, 5f, 10, false);
			TrackEventEngine engine = TrackEventEngine.create(trackConfig(),
				crossing, counter, dwell, direction);

			List<Map<String, Object>> frameViews = new ArrayList<>();
			List<Map<String, Object>> eventViews = new ArrayList<>();
			Map<String, Integer> counts = new LinkedHashMap<>();
			for (TrackEventType type : TrackEventType.values()) {
				counts.put(type.name(), 0);
			}
			int maxInside = 0;
			for (FrameRequest frame : request.getFrames()) {
				if (frame == null) {
					continue;
				}
				FrameResult result = engine.process(toBoxes(frame.getBoxes()), frame.getTimestampMs());
				List<Map<String, Object>> tracks = new ArrayList<>();
				for (Track track : result.getTracks()) {
					tracks.add(trackView(track));
				}
				Map<String, Object> frameView = new LinkedHashMap<>();
				frameView.put("frameIndex", result.getFrameIndex());
				frameView.put("timestampMs", result.getTimestampMs());
				frameView.put("tracks", tracks);
				frameView.put("eventCount", result.getEvents().size());
				frameViews.add(frameView);

				for (TrackEvent event : result.getEvents()) {
					eventViews.add(eventView(event));
					counts.put(event.getType().name(), counts.get(event.getType().name()) + 1);
				}
				maxInside = Math.max(maxInside, counter.getCurrentInside().size());
			}

			Map<String, Object> summary = new LinkedHashMap<>();
			summary.putAll(counts);
			summary.put("enters", counter.getEnterCount());
			summary.put("leaves", counter.getLeaveCount());
			summary.put("maxInside", maxInside);

			Map<String, Object> body = new LinkedHashMap<>();
			body.put("frameCount", frameViews.size());
			body.put("frames", frameViews);
			body.put("events", eventViews);
			body.put("summary", summary);
			return json(body);
		} catch (IllegalArgumentException e) {
			log.warn("逐帧事件分析失败: {}", e.getMessage());
			return error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}
}
