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
package net.dreamlu.mica.ai.example.config;

import net.dreamlu.mica.ai.example.pipeline.FacePipeline;
import net.dreamlu.mica.ai.example.repository.SimpleMemoryRepository;
import net.dreamlu.mica.ai.example.repository.VectorRepository;
import net.dreamlu.mica.ai.face.alignment.FaceAligner;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.liveness.LivenessDetector;
import net.dreamlu.mica.ai.face.recognition.FeatureExtractor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 人脸识别服务装配。
 * <p>
 * 提供一个最小可用的 {@link VectorRepository}（基于 {@link SimpleMemoryRepository}），
 * 再把 starter 暴露的 {@link FaceDetector} / {@link FaceAligner} / {@link FeatureExtractor}
 * / {@link LivenessDetector} 组装成 {@link FacePipeline}。
 * 生产环境请在自己的业务模块里换成 Milvus / Qdrant / Elasticsearch 等。
 * </p>
 */
@Configuration
public class FaceRecognitionConfig {

	@Bean
	@ConditionalOnMissingBean(VectorRepository.class)
	public VectorRepository vectorRepository() {
		return new SimpleMemoryRepository();
	}

	@Bean
	public FacePipeline facePipeline(FaceDetector faceDetector,
									 FaceAligner faceAligner,
									 FeatureExtractor featureExtractor,
									 @Autowired(required = false) LivenessDetector livenessDetector,
									 VectorRepository vectorRepository) {
		return new FacePipeline(faceDetector, faceAligner, featureExtractor, livenessDetector, vectorRepository);
	}
}
