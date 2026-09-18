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
package net.dreamlu.mica.ai.example.pipeline;

import lombok.Getter;
import net.dreamlu.mica.ai.face.alignment.FaceAligner;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.liveness.LivenessDetector;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.LivenessResult;
import net.dreamlu.mica.ai.face.model.MatchResult;
import net.dreamlu.mica.ai.face.recognition.FeatureExtractor;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import net.dreamlu.mica.ai.example.repository.VectorRepository;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 人脸识别管线编排示例。
 * <p>
 * 串联 检测 → 活体（可选）→ 对齐 → 特征提取 → 向量检索 的完整流程。
 * 检测到的人脸若未通过活体，将跳过特征提取与向量检索，对应字段填 {@code null} / 空列表。
 * </p>
 * <p>
 * 本类位于 example 模块，作为参考实现；业务模块可按需复制改造。
 * </p>
 */
@Getter
public class FacePipeline {

	private final FaceDetector detector;
	private final FaceAligner aligner;
	private final FeatureExtractor extractor;
	private final LivenessDetector livenessDetector;
	private final VectorRepository vectorRepository;

	public FacePipeline(FaceDetector detector,
						FaceAligner aligner,
						FeatureExtractor extractor,
						LivenessDetector livenessDetector,
						VectorRepository vectorRepository) {
		this.detector = detector;
		this.aligner = aligner;
		this.extractor = extractor;
		this.livenessDetector = livenessDetector;
		this.vectorRepository = vectorRepository;
	}

	/**
	 * 对单张图像执行完整识别。
	 *
	 * @param image BGR 格式的 OpenCV Mat
	 * @return 识别结果，调用方负责释放内部可能持有的 Mat
	 */
	public FaceRecognitionResult recognize(Mat image) {
		FaceRecognitionResult result = new FaceRecognitionResult();
		List<FaceBox> boxes = detector.detect(image);
		result.setFaces(boxes);

		int n = boxes.size();
		List<float[]> features = new ArrayList<>(n);
		List<List<MatchResult>> matches = new ArrayList<>(n);
		Map<Integer, LivenessResult> livenessMap = new HashMap<>();
		result.setFeatures(features);
		result.setMatches(matches);
		result.setLivenessMap(livenessMap);

		for (int i = 0; i < n; i++) {
			FaceBox box = boxes.get(i);
			// 1. 活体检测（可选）
			boolean live = true;
			if (livenessDetector != null) {
				LivenessResult lr = livenessDetector.check(image, box);
				livenessMap.put(i, lr);
				live = lr.isLive();
			}

			// 2. 对齐
			Mat aligned = null;
			try {
				aligned = aligner.align(image, box);

				// 3. 活体未通过时不提取特征、不做向量检索
				if (!live) {
					features.add(null);
					matches.add(Collections.emptyList());
					continue;
				}

				// 4. 特征提取 + 向量检索
				float[] feat = extractor.extract(aligned);
				features.add(feat);
				matches.add(vectorRepository == null
					? Collections.<MatchResult>emptyList()
					: vectorRepository.search(feat, 1));
			} finally {
				ImageUtils.releaseAll(aligned);
			}
		}

		return result;
	}
}
