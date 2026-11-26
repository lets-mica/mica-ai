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
import lombok.Getter;
import lombok.Setter;
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
import net.dreamlu.mica.ai.track.motion.AffineMotionModel;
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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 视频事件分析 · 相机运动补偿（CMC）场景。
 *
 * <p>球机云台转动、车载 / 无人机挂载摄像机移动时，静止目标的框在图像里整体位移，
 * 轨迹框与检测框的 IoU 塌陷 → 轨迹断裂、ID 换新 → 越线 / ROI 进出 / 滞留 / 折返事件全部失真。
 *
 * <p>本控制器演示 mica-ai-track 的运动补偿能力：
 * <ul>
 *   <li>{@code POST /scenario/video-events/analyze-with-motion} —— 同一份输入，
 *       「有补偿 vs 无补偿」并列输出，直接看到 ID 数与事件计数的差异</li>
 *   <li>{@code POST /scenario/video-events/motion-demo} —— 零输入合成演示：
 *       相机横摇 + 世界静止的目标，无补偿 ID 断裂、有补偿 ID 稳定</li>
 * </ul>
 *
 * <p>与 {@link VideoEventScenarioController} 的 {@code /analyze} 端点<b>并存</b>，
 * 旧端点签名不变、行为不变。
 */
@Slf4j
@Tag(name = "视频事件分析 · 相机运动补偿",
	description = "球机 / 车载摄像机运动补偿 · 轨迹 ID 稳定 · 有补偿 vs 无补偿对照")
@RestController
@RequestMapping("/scenario/video-events")
@ConditionalOnProperty(prefix = "mica.ai.track", name = "enabled", havingValue = "true", matchIfMissing = true)
public class VideoMotionScenarioController {

	/**
	 * 仿射增量系数：{@code x' = a*x + b*y + c}，{@code y' = d*x + e*y + f}。
	 *
	 * <p>表示「本帧相对上一帧」的相机运动，由调用方逐帧累加。
	 */
	@Data
	public static class AffineRequest {
		private float a = 1f;
		private float b = 0f;
		private float c = 0f;
		private float d = 0f;
		private float e = 1f;
		private float f = 0f;
	}

	/**
	 * 带相机运动的逐帧事件分析请求（继承普通分析请求，追加运动信息）。
	 */
	@Getter
	@Setter
	public static class AnalyzeWithMotionRequest extends VideoEventScenarioController.AnalyzeRequest {
		/**
		 * 每帧相对<b>上一帧</b>的相机运动增量；长度 = 帧数 - 1。
		 *
		 * <p>第 1 帧即<b>参考帧</b>（相机固定坐标系），从第 2 帧起每帧追加一次增量。
		 */
		private List<AffineRequest> increments = new ArrayList<>();
		/** 是否同时跑一遍「无补偿」的结果做对照 */
		private boolean compareWithoutMotion = true;
	}

	@Operation(summary = "带相机运动补偿的逐帧事件分析",
		description = "请求体在普通分析请求基础上追加每帧相机运动增量（仿射系数），"
			+ "返回有补偿的结果，并可选并列返回无补偿的结果做对照")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "分析成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class))),
		@ApiResponse(responseCode = "400", description = "请求体非法或时间戳倒退", content = @Content)
	})
	@PostMapping(value = "/analyze-with-motion", consumes = MediaType.APPLICATION_JSON_VALUE)
	public ResponseEntity<Object> analyzeWithMotion(@RequestBody AnalyzeWithMotionRequest request) {
		if (request == null || request.getFrames() == null || request.getFrames().isEmpty()) {
			return VideoEventScenarioController.error("frames 不能为空，至少需要一帧", HttpStatus.BAD_REQUEST);
		}
		List<AffineRequest> increments = request.getIncrements();
		if (increments != null && !increments.isEmpty()
			&& increments.size() != request.getFrames().size() - 1) {
			return VideoEventScenarioController.error(
				"increments 长度必须等于帧数 - 1（第 1 帧是参考帧，没有增量），实际="
					+ increments.size() + "，帧数=" + request.getFrames().size(), HttpStatus.BAD_REQUEST);
		}
		try {
			Map<String, Object> body = new LinkedHashMap<>();
			body.put("frameCount", request.getFrames().size());
			body.put("withMotion", run(request, increments, true));
			if (request.isCompareWithoutMotion()) {
				body.put("withoutMotion", run(request, null, false));
			}
			return VideoEventScenarioController.json(body);
		} catch (IllegalArgumentException e) {
			log.warn("带运动补偿的事件分析失败: {}", e.getMessage());
			return VideoEventScenarioController.error(e.getMessage(), HttpStatus.BAD_REQUEST);
		}
	}

	@Operation(summary = "相机运动补偿合成演示",
		description = "零输入：内置「相机横摇 + 世界静止目标」的合成序列，"
			+ "并列输出无补偿（轨迹断裂、ID 换新、ROI 重复计数）与有补偿（ID 稳定、计数正确）的结果")
	@ApiResponses(value = {
		@ApiResponse(responseCode = "200", description = "演示成功（JSON）",
			content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE,
				schema = @Schema(implementation = Map.class)))
	})
	@PostMapping(value = "/motion-demo", consumes = MediaType.APPLICATION_JSON_VALUE)
	public ResponseEntity<Object> motionDemo() {
		Map<String, Object> body = new LinkedHashMap<>();
		body.put("frameCount", DemoConfig.FRAMES);
		body.put("cameraMotion", demoCameraMotion());
		body.put("target", demoTarget());
		body.put("withoutMotion", runDemo(false));
		body.put("withMotion", runDemo(true));
		return VideoEventScenarioController.json(body);
	}

	/**
	 * 跑一遍逐帧分析。
	 *
	 * @param increments 每帧相机运动增量；{@code null} 或空表示不做补偿
	 * @param useMotion  是否启用相机运动补偿
	 */
	private Map<String, Object> run(VideoEventScenarioController.AnalyzeRequest base,
								   List<AffineRequest> increments, boolean useMotion) {
		VideoEventScenarioController.LineRequest line = base.getLine() == null
			? new VideoEventScenarioController.LineRequest() : base.getLine();
		VideoEventScenarioController.RoiRequest roi = base.getRoi() == null
			? new VideoEventScenarioController.RoiRequest() : base.getRoi();
		Roi zone = Roi.rectangle(roi.getX1(), roi.getY1(), roi.getX2(), roi.getY2());
		AnchorPolicy anchor = roi.isBottomCenterAnchor()
			? AnchorPolicy.BOTTOM_CENTER : AnchorPolicy.CENTER;
		RoiCounterRule counter = new RoiCounterRule(roi.getName(), zone, anchor, false);
		LineCrossingRule crossing = new LineCrossingRule(line.getName(),
			line.getX1(), line.getY1(), line.getX2(), line.getY2(),
			LineCrossingRule.Direction.BOTH, false, true);
		DwellRule dwell = new DwellRule(roi.getName(), zone,
			base.getDwellThresholdMs(), anchor, false);
		DirectionRule direction = new DirectionRule("u-turn", 120d, 5f, 10, false);

		AffineMotionModel motion = useMotion ? AffineMotionModel.identity() : null;
		TrackEventEngine engine = useMotion
			? TrackEventEngine.create(VideoEventScenarioController.trackConfig(), motion,
				crossing, counter, dwell, direction)
			: TrackEventEngine.create(VideoEventScenarioController.trackConfig(),
				crossing, counter, dwell, direction);

		List<Map<String, Object>> frameViews = new ArrayList<>();
		List<Map<String, Object>> eventViews = new ArrayList<>();
		Map<String, Integer> counts = new LinkedHashMap<>();
		for (TrackEventType type : TrackEventType.values()) {
			counts.put(type.name(), 0);
		}
		Set<Integer> trackIds = new LinkedHashSet<>();
		int maxInside = 0;

		List<VideoEventScenarioController.FrameRequest> frames = base.getFrames();
		for (int i = 0; i < frames.size(); i++) {
			VideoEventScenarioController.FrameRequest frame = frames.get(i);
			// 第 1 帧是参考帧，不追加增量；从第 2 帧起每帧追加一次「相对上一帧」的增量
			if (useMotion && i > 0 && increments != null && i - 1 < increments.size()) {
				AffineRequest increment = increments.get(i - 1);
				if (increment != null) {
					motion.append(AffineMotionModel.of(increment.getA(), increment.getB(), increment.getC(),
						increment.getD(), increment.getE(), increment.getF()));
				}
			}
			List<BBox> boxes = frame == null
				? new ArrayList<>() : VideoEventScenarioController.toBoxes(frame.getBoxes());
			long timestampMs = frame == null ? 0L : frame.getTimestampMs();

			FrameResult result = engine.process(boxes, timestampMs);
			List<Map<String, Object>> tracks = new ArrayList<>();
			for (Track track : result.getTracks()) {
				tracks.add(trackView(track));
				trackIds.add(track.getId());
			}
			Map<String, Object> frameView = new LinkedHashMap<>();
			frameView.put("frameIndex", result.getFrameIndex());
			frameView.put("timestampMs", result.getTimestampMs());
			frameView.put("tracks", tracks);
			frameView.put("eventCount", result.getEvents().size());
			frameViews.add(frameView);

			for (TrackEvent event : result.getEvents()) {
				eventViews.add(VideoEventScenarioController.eventView(event));
				counts.put(event.getType().name(), counts.get(event.getType().name()) + 1);
			}
			maxInside = Math.max(maxInside, counter.getCurrentInside().size());
		}

		Map<String, Object> summary = new LinkedHashMap<>();
		summary.putAll(counts);
		summary.put("enters", counter.getEnterCount());
		summary.put("leaves", counter.getLeaveCount());
		summary.put("maxInside", maxInside);
		summary.put("trackIdCount", trackIds.size());
		summary.put("trackIds", new ArrayList<>(trackIds));

		Map<String, Object> result = new LinkedHashMap<>();
		result.put("frames", frameViews);
		result.put("events", eventViews);
		result.put("summary", summary);
		return result;
	}

	/**
	 * 轨迹视图：额外带参考帧坐标（启用补偿时才有值）。
	 */
	private static Map<String, Object> trackView(Track track) {
		TrackPoint point = track.getLastPoint();
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("id", track.getId());
		m.put("classId", track.getClassId());
		m.put("cx", point == null ? 0f : point.getCx());
		m.put("cy", point == null ? 0f : point.getCy());
		m.put("refCx", point == null || !point.hasReference() ? null : point.getRefCx());
		m.put("refCy", point == null || !point.hasReference() ? null : point.getRefCy());
		m.put("direction", track.directionLabel());
		m.put("speedPxPerSec", track.speedPerSecond());
		m.put("age", track.getAge());
		m.put("lostFrames", track.getLostFrames());
		return m;
	}

	/**
	 * 合成演示的固定参数。
	 */
	private static final class DemoConfig {
		/** 帧数 */
		static final int FRAMES = 16;
		/** 帧间隔（毫秒） */
		static final long FRAME_INTERVAL_MS = 100L;
		/** 目标边长（像素） */
		static final float BOX_SIZE = 40f;
		/** 目标在第 1 帧的左上角 x */
		static final float START_X = 500f;
		/** 目标在第 1 帧的左上角 y */
		static final float START_Y = 280f;
		/** 相机每帧横摇导致的图像内容左移量（像素） */
		static final float DX_PER_FRAME = 20f;
		/** ROI 右边界：目标从 ROI 右侧进入 */
		static final float ROI_X2 = 400f;
		/** ROI 高度 */
		static final float ROI_Y2 = 600f;

		private DemoConfig() {
		}
	}

	/**
	 * 演示用跟踪配置：匹配阈值取 0.5（ByteTrack 常用值），目标逐帧位移 20px 时
	 * 相邻帧 IoU 只有 0.33 —— 无补偿必然断裂。
	 */
	private static ByteTrackConfig demoTrackConfig() {
		return ByteTrackConfig.builder()
			.highThreshold(0.5f)
			.lowThreshold(0.1f)
			.matchThresh(0.5f)
			.matchLowThresh(0.3f)
			.confirmAge(2)
			.trackBufferedThresh(30)
			.historySize(64)
			.build();
	}

	/**
	 * 合成演示：目标在世界坐标静止，相机每帧横摇 → 图像内容整体左移 {@code dx}。
	 */
	private static List<BBox> demoBoxes(int frame) {
		float x = DemoConfig.START_X - DemoConfig.DX_PER_FRAME * frame;
		float y = DemoConfig.START_Y;
		float size = DemoConfig.BOX_SIZE;
		List<BBox> boxes = new ArrayList<>();
		boxes.add(new BBox(x, y, x + size, y + size, 0.9f, 0));
		return boxes;
	}

	private static Map<String, Object> runDemo(boolean useMotion) {
		AffineMotionModel motion = useMotion ? AffineMotionModel.identity() : null;
		Roi zone = Roi.rectangle(0f, 0f, DemoConfig.ROI_X2, DemoConfig.ROI_Y2);
		RoiCounterRule counter = new RoiCounterRule("zone", zone, AnchorPolicy.CENTER, false);
		TrackEventEngine engine = useMotion
			? TrackEventEngine.create(demoTrackConfig(), motion, counter)
			: TrackEventEngine.create(demoTrackConfig(), counter);

		Set<Integer> trackIds = new LinkedHashSet<>();
		List<float[]> imagePath = new ArrayList<>();
		List<float[]> referencePath = new ArrayList<>();
		Track lastTrack = null;

		for (int frame = 0; frame < DemoConfig.FRAMES; frame++) {
			// 第 1 帧是参考帧，不追加增量；之后每帧追加一次「图像内容左移 dx」
			if (useMotion && frame > 0) {
				motion.append(AffineMotionModel.translation(-DemoConfig.DX_PER_FRAME, 0f));
			}
			FrameResult result = engine.process(demoBoxes(frame), frame * DemoConfig.FRAME_INTERVAL_MS);
			for (Track track : result.getTracks()) {
				trackIds.add(track.getId());
				lastTrack = track;
				TrackPoint point = track.getLastPoint();
				if (point == null) {
					continue;
				}
				imagePath.add(new float[]{point.getCx(), point.getCy()});
				if (point.hasReference()) {
					referencePath.add(new float[]{point.getRefCx(), point.getRefCy()});
				}
			}
		}

		Map<String, Object> result = new LinkedHashMap<>();
		result.put("trackIdCount", trackIds.size());
		result.put("trackIds", new ArrayList<>(trackIds));
		result.put("roiEnters", counter.getEnterCount());
		result.put("roiLeaves", counter.getLeaveCount());
		// 表观速度：图像坐标下目标位移 ÷ 时间 —— 相机一动它就很大
		result.put("apparentSpeedPxPerSec", speedOf(imagePath));
		// 参考帧速度：世界坐标下目标位移 ÷ 时间 —— 目标静止就是 0
		result.put("referenceSpeedPxPerSec", referencePath.isEmpty() ? null : speedOf(referencePath));
		// 跟踪器报告的速度：轨迹断裂成单点时无速度可言
		result.put("reportedSpeedPxPerSec",
			lastTrack != null && lastTrack.getPoints().size() >= 2 ? lastTrack.speedPerSecond() : null);
		result.put("imagePath", imagePath);
		result.put("referencePath", referencePath.isEmpty() ? null : referencePath);
		return result;
	}

	/**
	 * 折线首尾位移 ÷ 帧间隔总和（像素 / 秒）。
	 */
	private static double speedOf(List<float[]> path) {
		if (path.size() < 2) {
			return 0d;
		}
		float[] first = path.get(0);
		float[] last = path.get(path.size() - 1);
		double seconds = (DemoConfig.FRAMES - 1) * DemoConfig.FRAME_INTERVAL_MS / 1000d;
		if (seconds <= 0d) {
			return 0d;
		}
		return Math.round((last[0] - first[0]) / seconds * 100d) / 100d;
	}

	private static Map<String, Object> demoCameraMotion() {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("dxPerFrame", -DemoConfig.DX_PER_FRAME);
		m.put("dyPerFrame", 0f);
		m.put("description", "相机横摇，图像内容整体左移 " + DemoConfig.DX_PER_FRAME + " px/帧；"
			+ "目标在世界坐标静止");
		return m;
	}

	private static Map<String, Object> demoTarget() {
		Map<String, Object> m = new LinkedHashMap<>();
		m.put("startX", DemoConfig.START_X);
		m.put("startY", DemoConfig.START_Y);
		m.put("size", DemoConfig.BOX_SIZE);
		m.put("description", "目标在世界坐标静止，图像坐标下逐帧左移");
		return m;
	}

}
