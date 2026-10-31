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

import net.dreamlu.mica.ai.classify.core.ClassifyEngine;
import net.dreamlu.mica.ai.docalign.core.DocAlignEngine;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.example.pipeline.document.DocumentPipeline;
import net.dreamlu.mica.ai.example.pipeline.idphoto.IdPhotoPipeline;
import net.dreamlu.mica.ai.face.alignment.FaceAligner;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.quality.FaceQualityAssessor;
import net.dreamlu.mica.ai.humanseg.core.HumanSegEngine;
import net.dreamlu.mica.ai.layout.pipeline.LayoutPipeline;
import net.dreamlu.mica.ai.matting.core.MattingEngine;
import net.dreamlu.mica.ai.textline.core.TextLineEngine;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 组合场景流水线的装配。
 *
 * <p>两条流水线都是「把若干能力的门面类编排起来」，本身不持有模型；
 * 所有依赖都来自各能力 starter 的自动装配。
 *
 * <p>依赖一律用 {@link ObjectProvider#getIfUnique()} 取：某个能力被
 * {@code enabled=false} 关掉时对应 Bean 根本不存在，此时流水线里就是 null，
 * 对应阶段自动降级跳过，而不是让整个应用启动失败。
 *
 * @author DreamLu
 */
@Configuration
public class ScenarioConfig {

	/**
	 * 智能文档流水线。
	 */
	@Bean
	@ConditionalOnMissingBean
	public DocumentPipeline documentPipeline(ObjectProvider<DocAlignEngine> docAlignEngine,
											 ObjectProvider<ClassifyEngine> classifyEngine,
											 ClassifyModelRegistry classifyRegistry,
											 ObjectProvider<EnhanceEngine> enhanceEngine,
											 ObjectProvider<LayoutPipeline> layoutPipeline,
											 ObjectProvider<TextLineEngine> textLineEngine) {
		return new DocumentPipeline(
			docAlignEngine.getIfUnique(),
			classifyEngine.getIfUnique(),
			classifyRegistry,
			enhanceEngine.getIfUnique(),
			layoutPipeline.getIfUnique(),
			textLineEngine.getIfUnique());
	}

	/**
	 * 证件照一键生成流水线。
	 */
	@Bean
	@ConditionalOnMissingBean
	public IdPhotoPipeline idPhotoPipeline(FaceDetector faceDetector,
										   FaceAligner faceAligner,
										   ObjectProvider<FaceQualityAssessor> qualityAssessor,
										   ObjectProvider<MattingEngine> mattingEngine,
										   ObjectProvider<HumanSegEngine> humanSegEngine,
										   ObjectProvider<EnhanceEngine> enhanceEngine) {
		return new IdPhotoPipeline(
			faceDetector,
			faceAligner,
			qualityAssessor.getIfUnique(),
			mattingEngine.getIfUnique(),
			humanSegEngine.getIfUnique(),
			enhanceEngine.getIfUnique());
	}

}
