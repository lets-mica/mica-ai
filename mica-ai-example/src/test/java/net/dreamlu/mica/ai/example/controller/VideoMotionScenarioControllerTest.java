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

import org.junit.jupiter.api.Test;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * 相机运动补偿场景端点测试。
 */
class VideoMotionScenarioControllerTest {

	private final VideoMotionScenarioController controller = new VideoMotionScenarioController();

	private static final float DX = 20f;
	private static final int FRAMES = 16;
	/** motion-demo 内置合成序列的目标边长 */
	private static final float DEMO_BOX_SIZE = 40f;
	/** analyze-with-motion 用例的目标边长：小目标在 20px 位移下 IoU 直接塌到 0.09 */
	private static final float BOX_SIZE = 24f;
	private static final float START_X = 500f;
	private static final float START_Y = 280f;

	/**
	 * 相机每帧横摇、目标在世界静止 → 图像坐标下目标逐帧左移 dx。
	 */
	private static List<VideoEventScenarioController.FrameRequest> frames() {
		List<VideoEventScenarioController.FrameRequest> frames = new ArrayList<>();
		for (int i = 0; i < FRAMES; i++) {
			float x = START_X - DX * i;
			VideoEventScenarioController.BoxRequest box = new VideoEventScenarioController.BoxRequest();
			box.setX1(x);
			box.setY1(START_Y);
			box.setX2(x + BOX_SIZE);
			box.setY2(START_Y + BOX_SIZE);
			box.setScore(0.9f);
			List<VideoEventScenarioController.BoxRequest> boxes = new ArrayList<>();
			boxes.add(box);
			VideoEventScenarioController.FrameRequest frame = new VideoEventScenarioController.FrameRequest();
			frame.setTimestampMs(i * 100L);
			frame.setBoxes(boxes);
			frames.add(frame);
		}
		return frames;
	}

	/**
	 * 每帧相对上一帧的相机运动增量：图像内容左移 dx。
	 */
	private static List<VideoMotionScenarioController.AffineRequest> increments() {
		List<VideoMotionScenarioController.AffineRequest> list = new ArrayList<>();
		for (int i = 1; i < FRAMES; i++) {
			VideoMotionScenarioController.AffineRequest affine = new VideoMotionScenarioController.AffineRequest();
			affine.setC(-DX);
			list.add(affine);
		}
		return list;
	}

	@SuppressWarnings("unchecked")
	@Test
	void motionDemoBreaksTrackWithoutCompensationAndKeepsItWithCompensation() {
		ResponseEntity<Object> response = controller.motionDemo();
		assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);

		Map<String, Object> body = (Map<String, Object>) response.getBody();
		Map<String, Object> without = (Map<String, Object>) body.get("withoutMotion");
		Map<String, Object> with = (Map<String, Object>) body.get("withMotion");

		// 无补偿：相机一动，轨迹框与检测框错位 → 每帧换新 ID
		assertThat(without.get("trackIdCount")).isEqualTo(FRAMES);
		// 有补偿：检测框变换回参考帧再匹配 → 轨迹 ID 稳定
		assertThat(with.get("trackIdCount")).isEqualTo(1);
	}

	@SuppressWarnings("unchecked")
	@Test
	void motionDemoReferencePathStaysFixedWhileImagePathDrifts() {
		ResponseEntity<Object> response = controller.motionDemo();
		Map<String, Object> body = (Map<String, Object>) response.getBody();
		Map<String, Object> with = (Map<String, Object>) body.get("withMotion");

		List<float[]> imagePath = (List<float[]>) with.get("imagePath");
		List<float[]> referencePath = (List<float[]>) with.get("referencePath");

		// 图像坐标下目标逐帧左移
		assertThat(imagePath).hasSize(FRAMES);
		assertThat(imagePath.get(0)[0]).isGreaterThan(imagePath.get(FRAMES - 1)[0]);
		// 参考帧（相机固定系）下目标静止
		assertThat(referencePath).hasSize(FRAMES);
		for (float[] point : referencePath) {
			assertThat(point[0]).isEqualTo(START_X + DEMO_BOX_SIZE / 2f);
		}
	}

	@SuppressWarnings("unchecked")
	@Test
	void motionDemoSpeedIsWorldMotionWithCompensation() {
		ResponseEntity<Object> response = controller.motionDemo();
		Map<String, Object> body = (Map<String, Object>) response.getBody();

		Map<String, Object> without = (Map<String, Object>) body.get("withoutMotion");
		Map<String, Object> with = (Map<String, Object>) body.get("withMotion");

		// 表观速度：相机在动，图像坐标下目标位移很大（-20px / 100ms）
		assertThat((Double) without.get("apparentSpeedPxPerSec")).isEqualTo(-200d);
		assertThat((Double) with.get("apparentSpeedPxPerSec")).isEqualTo(-200d);
		// 有补偿：参考帧（世界坐标）下目标静止 → 真实速度为 0
		assertThat((Double) with.get("referenceSpeedPxPerSec")).isEqualTo(0d);
		assertThat((Double) with.get("reportedSpeedPxPerSec")).isEqualTo(0d);
		// 无补偿：轨迹断裂成单点，跟踪器根本没有速度可言
		assertThat(without.get("reportedSpeedPxPerSec")).isNull();
	}

	@SuppressWarnings("unchecked")
	@Test
	void motionDemoRoiCounterCountsOnceWithCompensation() {
		ResponseEntity<Object> response = controller.motionDemo();
		Map<String, Object> body = (Map<String, Object>) response.getBody();

		Map<String, Object> without = (Map<String, Object>) body.get("withoutMotion");
		Map<String, Object> with = (Map<String, Object>) body.get("withMotion");

		// 无补偿：ID 换新 → 同一个目标被反复计成「进入」
		assertThat((Integer) without.get("roiEnters")).isGreaterThan(1);
		// 有补偿：同一目标只进一次
		assertThat((Integer) with.get("roiEnters")).isEqualTo(1);
	}

	@SuppressWarnings("unchecked")
	@Test
	void analyzeWithMotionKeepsTrackIdStable() {
		VideoMotionScenarioController.AnalyzeWithMotionRequest request =
			new VideoMotionScenarioController.AnalyzeWithMotionRequest();
		request.setFrames(frames());
		request.setIncrements(increments());
		request.setCompareWithoutMotion(true);

		ResponseEntity<Object> response = controller.analyzeWithMotion(request);
		assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);

		Map<String, Object> body = (Map<String, Object>) response.getBody();
		Map<String, Object> with = (Map<String, Object>) body.get("withMotion");
		Map<String, Object> without = (Map<String, Object>) body.get("withoutMotion");

		Map<String, Object> withSummary = (Map<String, Object>) with.get("summary");
		Map<String, Object> withoutSummary = (Map<String, Object>) without.get("summary");

		assertThat(withSummary.get("trackIdCount")).isEqualTo(1);
		assertThat(withoutSummary.get("trackIdCount")).isEqualTo(FRAMES);
	}

	@Test
	void analyzeWithMotionRejectsMismatchedIncrementLength() {
		VideoMotionScenarioController.AnalyzeWithMotionRequest request =
			new VideoMotionScenarioController.AnalyzeWithMotionRequest();
		request.setFrames(frames());
		request.setIncrements(new ArrayList<>());
		request.getIncrements().add(new VideoMotionScenarioController.AffineRequest());

		ResponseEntity<Object> response = controller.analyzeWithMotion(request);
		assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
	}

	@Test
	void analyzeWithMotionRejectsEmptyFrames() {
		VideoMotionScenarioController.AnalyzeWithMotionRequest request =
			new VideoMotionScenarioController.AnalyzeWithMotionRequest();

		ResponseEntity<Object> response = controller.analyzeWithMotion(request);
		assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
	}

}
