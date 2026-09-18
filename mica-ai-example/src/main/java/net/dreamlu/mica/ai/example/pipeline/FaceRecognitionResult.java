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

import lombok.Data;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.LivenessResult;
import net.dreamlu.mica.ai.face.model.MatchResult;

import java.util.List;
import java.util.Map;

/**
 * 人脸识别整体结果。
 * <p>
 * 一次识别请求中可能包含多张人脸，每个列表的下标一一对应。
 * </p>
 */
@Data
public class FaceRecognitionResult {

	/**
	 * 检测到的人脸框列表
	 */
	private List<FaceBox> faces;

	/**
	 * 每张人脸对应的特征向量（{@link net.dreamlu.mica.ai.face.recognition.FeatureExtractor#FEATURE_DIM}
	 * 维 float 数组，SFace 为 128 维），若对应人脸活体未通过则为 {@code null}
	 */
	private List<float[]> features;

	/**
	 * 每张人脸对应的 TopK 匹配结果，列表与 {@link #faces} 等长
	 */
	private List<List<MatchResult>> matches;

	/**
	 * 活体检测结果，key 为人脸下标。仅当启用活体检测时填充。
	 */
	private Map<Integer, LivenessResult> livenessMap;
}
