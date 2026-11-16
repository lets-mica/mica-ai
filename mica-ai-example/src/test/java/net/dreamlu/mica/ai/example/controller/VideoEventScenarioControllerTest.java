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

class VideoEventScenarioControllerTest {

	private final VideoEventScenarioController controller = new VideoEventScenarioController();

	private static VideoEventScenarioController.BoxRequest box(float cx) {
		VideoEventScenarioController.BoxRequest b = new VideoEventScenarioController.BoxRequest();
		b.setX1(cx - 30f);
		b.setY1(20f);
		b.setX2(cx + 30f);
		b.setY2(80f);
		b.setScore(0.9f);
		b.setClassId(7);
		return b;
	}

	private static VideoEventScenarioController.AnalyzeRequest request(boolean descending) {
		VideoEventScenarioController.AnalyzeRequest request = new VideoEventScenarioController.AnalyzeRequest();
		List<VideoEventScenarioController.FrameRequest> frames = new ArrayList<>();
		float[] centers = {20f, 40f, 60f, 80f, 100f, 120f};
		for (int i = 0; i < centers.length; i++) {
			VideoEventScenarioController.FrameRequest frame = new VideoEventScenarioController.FrameRequest();
			frame.setTimestampMs(descending && i == 5 ? 100L : i * 300L);
			List<VideoEventScenarioController.BoxRequest> boxes = new ArrayList<>();
			boxes.add(box(centers[i]));
			frame.setBoxes(boxes);
			frames.add(frame);
		}
		request.setFrames(frames);
		request.setDwellThresholdMs(1000L);
		return request;
	}

	@SuppressWarnings("unchecked")
	@Test
	void analyzeProducesCrossingDwellAndRoiEvents() {
		ResponseEntity<Object> response = controller.analyze(request(false));
		assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);

		Map<String, Object> body = (Map<String, Object>) response.getBody();
		assertThat(body).isNotNull();
		assertThat(body.get("frameCount")).isEqualTo(6);
		List<Map<String, Object>> frames = (List<Map<String, Object>>) body.get("frames");
		assertThat(frames).hasSize(6);
		assertThat(frames.get(0).get("frameIndex")).isEqualTo(1);
		assertThat(frames.get(5).get("timestampMs")).isEqualTo(1500L);

		List<Map<String, Object>> events = (List<Map<String, Object>>) body.get("events");
		assertThat(events).extracting(e -> e.get("type"))
			.contains("ROI_ENTERED", "DWELL_EXCEEDED", "LINE_CROSSED")
			.doesNotContain("DIRECTION_CHANGED");
		assertThat(events).filteredOn(e -> "LINE_CROSSED".equals(e.get("type")))
			.singleElement().satisfies(e -> assertThat(e.get("direction")).isEqualTo("BACKWARD"));

		Map<String, Object> summary = (Map<String, Object>) body.get("summary");
		assertThat(summary.get("LINE_CROSSED")).isEqualTo(1);
		assertThat(summary.get("DWELL_EXCEEDED")).isEqualTo(1);
		assertThat(summary.get("enters")).isEqualTo(1);
		assertThat(summary.get("maxInside")).isEqualTo(1);
	}

	@Test
	void emptyFramesReturnsBadRequest() {
		VideoEventScenarioController.AnalyzeRequest empty = new VideoEventScenarioController.AnalyzeRequest();
		ResponseEntity<Object> response = controller.analyze(empty);
		assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
	}

	@Test
	void descendingTimestampReturnsBadRequest() {
		ResponseEntity<Object> response = controller.analyze(request(true));
		assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
	}
}
