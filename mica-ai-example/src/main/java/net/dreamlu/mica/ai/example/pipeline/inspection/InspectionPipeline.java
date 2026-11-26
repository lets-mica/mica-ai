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
package net.dreamlu.mica.ai.example.pipeline.inspection;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.detect.core.DetectSource;
import net.dreamlu.mica.ai.detect.core.SlicedDetectEngine;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import net.dreamlu.mica.ai.detect.util.DetectImageUtils;
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
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 小球机巡检管线：切片检测 → 运动补偿跟踪 → 行为事件。
 *
 * <p>三段职责各自独立、缺一不可：
 * <ol>
 *   <li><b>看得见</b>——{@code mica-ai-detect} 的切片推理（{@link SlicedDetectEngine}）：
 *       远处小目标在整图 letterbox 后不足 20px，整图检不出，切片按原像素推理可检出</li>
 *   <li><b>跟得住</b>——{@code mica-ai-track} 的运动补偿（{@link AffineMotionModel}）：
 *       相机横摇时轨迹框与检测框不再错位，ID 不断裂</li>
 *   <li><b>判得准</b>——{@code mica-ai-track} 的行为规则：越线 / ROI 进出 / 滞留 / 折返</li>
 * </ol>
 *
 * <p>本类位于 example 模块，作为参考实现；业务模块可按需复制改造。
 */
@Slf4j
public class InspectionPipeline {

	/** 检测能力（整图推理；生产实现是 {@code DetectEngine}，测试可换假实现） */
	private final DetectSource detectSource;

	/** 切片检测引擎 */
	private final SlicedDetectEngine slicedDetectEngine;

	/** 类别标签集（与检测模型强耦合） */
	private final String[] labels;

	/** 跟踪配置 */
	private final ByteTrackConfig trackConfig;

	public InspectionPipeline(DetectSource detectSource, SlicedDetectEngine slicedDetectEngine,
							  String[] labels) {
		this(detectSource, slicedDetectEngine, labels, ByteTrackConfig.builder()
			.highThreshold(0.5f)
			.lowThreshold(0.1f)
			.matchThresh(0.5f)
			.matchLowThresh(0.3f)
			.confirmAge(2)
			.trackBufferedThresh(30)
			.historySize(64)
			.build());
	}

	public InspectionPipeline(DetectSource detectSource, SlicedDetectEngine slicedDetectEngine,
							  String[] labels, ByteTrackConfig trackConfig) {
		this.detectSource = detectSource;
		this.slicedDetectEngine = slicedDetectEngine;
		this.labels = labels;
		this.trackConfig = trackConfig;
	}

	/**
	 * 跑完整条巡检管线。
	 *
	 * @param frameBytes 各帧图片字节（按时间顺序）
	 * @param options    阶段开关与规则参数
	 * @return 巡检结果
	 */
	public InspectionResult process(List<byte[]> frameBytes, InspectionOptions options) {
		InspectionOptions opt = options == null ? new InspectionOptions() : options;

		Roi zone = Roi.rectangle(opt.getRoi()[0], opt.getRoi()[1], opt.getRoi()[2], opt.getRoi()[3]);
		AnchorPolicy anchor = opt.isBottomCenterAnchor() ? AnchorPolicy.BOTTOM_CENTER : AnchorPolicy.CENTER;
		RoiCounterRule counter = new RoiCounterRule("zone", zone, anchor, false);
		LineCrossingRule crossing = new LineCrossingRule("gate",
			opt.getLine()[0], opt.getLine()[1], opt.getLine()[2], opt.getLine()[3],
			LineCrossingRule.Direction.BOTH, false, true);
		DwellRule dwell = new DwellRule("zone", zone, opt.getDwellThresholdMs(), anchor, false);
		DirectionRule direction = new DirectionRule("u-turn", 120d, 5f, 10, false);

		AffineMotionModel motion = opt.isMotionCompensation() ? AffineMotionModel.identity() : null;
		TrackEventEngine engine = motion == null
			? TrackEventEngine.create(trackConfig, crossing, counter, dwell, direction)
			: TrackEventEngine.create(trackConfig, motion, crossing, counter, dwell, direction);

		String[] labels = this.labels;
		List<InspectionResult.FrameReport> frameReports = new ArrayList<>();
		List<TrackEvent> allEvents = new ArrayList<>();
		Set<Integer> trackIds = new LinkedHashSet<>();
		Map<String, Integer> counts = new LinkedHashMap<>();
		for (TrackEventType type : TrackEventType.values()) {
			counts.put(type.name(), 0);
		}
		int maxInside = 0;
		byte[] annotatedPng = null;

		for (int i = 0; i < frameBytes.size(); i++) {
			byte[] bytes = frameBytes.get(i);
			Mat image = null;
			try {
				image = DetectImageUtils.byteArrayToMat(bytes);

				// ---- 阶段 1：检测（切片 / 整图） ----
				DetectResult detectResult = opt.isSliced()
					? slicedDetectEngine.detect(image) : detectSource.detect(image);
				int sliceCount = opt.isSliced()
					? slicedDetectEngine.sliceCount(detectResult.getWidth(), detectResult.getHeight()) : 1;

				// ---- 阶段 2：运动补偿（参考帧 = 第 1 帧，之后每帧追加一次增量） ----
				if (motion != null && i > 0) {
					float[] increment = incrementAt(opt, i);
					if (increment != null) {
						motion.append(AffineMotionModel.of(increment[0], increment[1], increment[2],
							increment[3], increment[4], increment[5]));
					}
				}

				// ---- 阶段 3：跟踪 + 事件规则 ----
				FrameResult result = engine.process(toBoxes(detectResult),
					i * opt.getFrameIntervalMs());

				List<InspectionResult.TrackView> trackViews = new ArrayList<>();
				for (Track track : result.getTracks()) {
					trackIds.add(track.getId());
					trackViews.add(trackView(track, labels));
				}
				List<TrackEvent> frameEvents = new ArrayList<>(result.getEvents());
				for (TrackEvent event : frameEvents) {
					allEvents.add(event);
					counts.put(event.getType().name(), counts.get(event.getType().name()) + 1);
				}
				maxInside = Math.max(maxInside, counter.getCurrentInside().size());

				frameReports.add(new InspectionResult.FrameReport(result.getFrameIndex(),
					result.getTimestampMs(), sliceCount, detectResult.size(), trackViews, frameEvents));

				// 最后一帧画标注图
				if (i == frameBytes.size() - 1) {
					annotatedPng = annotate(image, result.getTracks(), labels);
				}
			} finally {
				if (image != null) {
					image.release();
				}
			}
		}

		Map<String, Object> summary = new LinkedHashMap<>();
		summary.putAll(counts);
		summary.put("enters", counter.getEnterCount());
		summary.put("leaves", counter.getLeaveCount());
		summary.put("maxInside", maxInside);
		summary.put("trackIdCount", trackIds.size());
		summary.put("trackIds", new ArrayList<>(trackIds));

		return new InspectionResult(frameReports, allEvents, summary, annotatedPng);
	}

	/**
	 * 取第 {@code frame} 帧对应的相机运动增量（{@code frame > 0}）。
	 */
	private static float[] incrementAt(InspectionOptions options, int frame) {
		List<float[]> increments = options.getMotionIncrements();
		if (increments == null || frame - 1 >= increments.size()) {
			return null;
		}
		return increments.get(frame - 1);
	}

	private static List<BBox> toBoxes(DetectResult result) {
		List<BBox> boxes = new ArrayList<>();
		for (DetectBox box : result.getBoxes()) {
			boxes.add(new BBox(box.getX1(), box.getY1(), box.getX2(), box.getY2(),
				box.getScore(), box.getClassId()));
		}
		return boxes;
	}

	private static InspectionResult.TrackView trackView(Track track, String[] labels) {
		TrackPoint point = track.getLastPoint();
		Float refCx = point == null || !point.hasReference() ? null : point.getRefCx();
		Float refCy = point == null || !point.hasReference() ? null : point.getRefCy();
		return new InspectionResult.TrackView(track.getId(), track.getClassId(),
			track.getClassId() >= 0 && labels != null && track.getClassId() < labels.length
				? labels[track.getClassId()] : String.valueOf(track.getClassId()),
			track.getScore(),
			point == null ? 0f : point.getCx(),
			point == null ? 0f : point.getCy(),
			refCx, refCy, track.speedPerSecond(), track.directionLabel(), track.getAge());
	}

	/**
	 * 在图上画出轨迹框、id 与轨迹点连线，输出 PNG。
	 */
	private static byte[] annotate(Mat image, List<Track> tracks, String[] labels) {
		Mat canvas = image.clone();
		MatOfByte buf = null;
		try {
			for (Track track : tracks) {
				BBox box = track.getLastBBox();
				Point p1 = new Point(Math.round(box.getX1()), Math.round(box.getY1()));
				Point p2 = new Point(Math.round(box.getX2()), Math.round(box.getY2()));
				Scalar color = new Scalar(0, 255, 0);
				Imgproc.rectangle(canvas, p1, p2, color, 2);
				String text = "id=" + track.getId();
				if (track.getClassId() >= 0 && labels != null && track.getClassId() < labels.length) {
					text = labels[track.getClassId()] + " " + text;
				}
				Imgproc.putText(canvas, text, p1, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, color, 2);

				// 轨迹点连线（图像坐标）
				List<TrackPoint> points = track.getPoints();
				for (int i = 1; i < points.size(); i++) {
					TrackPoint from = points.get(i - 1);
					TrackPoint to = points.get(i);
					Imgproc.line(canvas,
						new Point(Math.round(from.getCx()), Math.round(from.getCy())),
						new Point(Math.round(to.getCx()), Math.round(to.getCy())),
						new Scalar(255, 128, 0), 2);
				}
			}
			buf = new MatOfByte();
			Imgcodecs.imencode(".png", canvas, buf);
			return buf.toArray();
		} finally {
			if (buf != null) {
				buf.release();
			}
			canvas.release();
		}
	}

}
