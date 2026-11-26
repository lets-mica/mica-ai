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

import net.dreamlu.mica.ai.detect.config.DetectConfig;
import net.dreamlu.mica.ai.detect.config.DetectModelFamily;
import net.dreamlu.mica.ai.detect.config.SlicedDetectConfig;
import net.dreamlu.mica.ai.detect.core.DetectSource;
import net.dreamlu.mica.ai.detect.core.SlicedDetectEngine;
import net.dreamlu.mica.ai.detect.model.DetectBox;
import net.dreamlu.mica.ai.detect.model.DetectResult;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 巡检管线测试：切片检测 → 运动补偿跟踪 → 行为事件。
 *
 * <p>用假检测源替代 ONNX 推理，只验证<b>三段能力如何协同</b>（检测框怎么进跟踪器、
 * 相机运动怎么被补偿、规则怎么被触发），检测模型本身的精度不在本类覆盖范围。
 */
class InspectionPipelineTest {

	private static final int FRAMES = 8;
	private static final float BOX_SIZE = 40f;
	private static final float START_X = 300f;
	private static final float BOX_Y = 100f;
	/** 相机每帧横摇导致的图像内容左移量 */
	private static final float DX = 20f;

	@BeforeAll
	static void setUp() {
		nu.pattern.OpenCV.loadLocally();
	}

	/**
	 * 假检测源：每次调用返回一个逐帧左移的框（模拟相机横摇 + 目标在世界静止）。
	 */
	private static final class MovingDetector implements DetectSource {

		private int call;

		@Override
		public DetectResult detect(Mat bgr) {
			float x = START_X - DX * call++;
			return new DetectResult(bgr.cols(), bgr.rows(),
				Collections.singletonList(new DetectBox(x, BOX_Y, x + BOX_SIZE, BOX_Y + BOX_SIZE, 0.9f, 0)));
		}
	}

	/**
	 * 固定位置的假检测源（切片模式下每片都会被调用一次）。
	 */
	private static final class FixedDetector implements DetectSource {

		@Override
		public DetectResult detect(Mat bgr) {
			return new DetectResult(bgr.cols(), bgr.rows(),
				Collections.singletonList(new DetectBox(300f, BOX_Y, 300f + BOX_SIZE, BOX_Y + BOX_SIZE, 0.9f, 0)));
		}
	}

	private static SlicedDetectConfig slicedConfig() {
		DetectConfig detectConfig = DetectConfig.builder()
			.family(DetectModelFamily.NANODET_PLUS)
			.modelPath("dummy-model.onnx")
			.inputSize(416)
			.build();
		return SlicedDetectConfig.builder()
			.detectConfig(detectConfig)
			.overlapRatio(0.25f)
			.build();
	}

	private static List<byte[]> frames(int width, int height, int count) {
		List<byte[]> frames = new ArrayList<>();
		for (int i = 0; i < count; i++) {
			Mat mat = new Mat(height, width, CvType.CV_8UC3, new Scalar(30, 30, 30));
			MatOfByte buf = new MatOfByte();
			try {
				Imgcodecs.imencode(".png", mat, buf);
				frames.add(buf.toArray());
			} finally {
				buf.release();
				mat.release();
			}
		}
		return frames;
	}

	private static InspectionOptions options(boolean motionCompensation) {
		InspectionOptions options = new InspectionOptions();
		options.setSliced(false);
		options.setMotionCompensation(motionCompensation);
		options.setRoi(new float[]{0f, 0f, 250f, 300f});
		options.setLine(new float[]{100f, 0f, 100f, 400f});
		options.setDwellThresholdMs(1000L);
		options.setFrameIntervalMs(100L);
		List<float[]> increments = new ArrayList<>();
		for (int i = 1; i < FRAMES; i++) {
			// 图像内容左移 dx：x' = x - dx
			increments.add(new float[]{1f, 0f, -DX, 0f, 1f, 0f});
		}
		options.setMotionIncrements(increments);
		return options;
	}

	@SuppressWarnings("unchecked")
	@Test
	void motionCompensationKeepsTrackIdStableAcrossFrames() {
		InspectionPipeline pipeline = new InspectionPipeline(
			new MovingDetector(), null, new String[]{"person"});

		InspectionResult result = pipeline.process(frames(400, 300, FRAMES), options(true));

		Map<String, Object> summary = result.getSummary();
		// 相机一动，检测框逐帧位移 20px（IoU 0.33 < 0.5）→ 无补偿必然每帧换 ID
		assertThat(summary.get("trackIdCount")).isEqualTo(1);
		// 同一个目标只进 ROI 一次
		assertThat(summary.get("enters")).isEqualTo(1);
	}

	@SuppressWarnings("unchecked")
	@Test
	void withoutCompensationTrackIdBreaksAndRoiCountsRepeat() {
		InspectionPipeline pipeline = new InspectionPipeline(
			new MovingDetector(), null, new String[]{"person"});

		InspectionResult result = pipeline.process(frames(400, 300, FRAMES), options(false));

		Map<String, Object> summary = result.getSummary();
		assertThat(summary.get("trackIdCount")).isEqualTo(FRAMES);
		assertThat((Integer) summary.get("enters")).isGreaterThan(1);
	}

	@Test
	void tracksCarryImageAndReferenceCoordinates() {
		InspectionPipeline pipeline = new InspectionPipeline(
			new MovingDetector(), null, new String[]{"person"});

		InspectionResult result = pipeline.process(frames(400, 300, FRAMES), options(true));

		InspectionResult.TrackView last = result.getFrames()
			.get(result.getFrames().size() - 1).getTracks().get(0);
		// 图像坐标：目标逐帧左移
		assertThat(last.getCx()).isLessThan(START_X);
		// 参考帧：目标在世界静止
		assertThat(last.getRefCx()).isEqualTo(START_X + BOX_SIZE / 2f);
		assertThat(last.getRefCy()).isEqualTo(BOX_Y + BOX_SIZE / 2f);
	}

	@Test
	void pipelineProducesAnnotatedPng() {
		InspectionPipeline pipeline = new InspectionPipeline(
			new MovingDetector(), null, new String[]{"person"});

		InspectionResult result = pipeline.process(frames(400, 300, FRAMES), options(true));

		assertThat(result.getAnnotatedPng()).isNotNull().isNotEmpty();
	}

	@Test
	void slicedModeReportsSliceCount() {
		FixedDetector detector = new FixedDetector();
		SlicedDetectEngine sliced = SlicedDetectEngine.create(detector, slicedConfig());
		InspectionPipeline pipeline = new InspectionPipeline(detector, sliced, new String[]{"person"});

		InspectionOptions options = options(true);
		options.setSliced(true);
		InspectionResult result = pipeline.process(frames(900, 700, 2), options);

		InspectionResult.FrameReport frame = result.getFrames().get(0);
		// 900×700 按 416 边长、25% 重叠切成多片
		assertThat(frame.getSliceCount()).isGreaterThan(1);
		// 假检测源不知道切片原点，每片都吐同一个绝对坐标的框 → 合并前每片各一个，
		// 跨片合并的正确性由 detect 模块自己的单测覆盖
		assertThat(frame.getDetectCount()).isEqualTo(frame.getSliceCount());
	}

}
