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
package net.dreamlu.mica.ai.example.enhance;

import net.dreamlu.mica.ai.enhance.config.EnhanceConfig;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.enhance.util.EnhanceImageUtils;
import net.dreamlu.mica.ai.example.config.EnhanceEnginePool;
import net.dreamlu.mica.ai.example.pipeline.enhance.EnhanceUpscaleOptions;
import net.dreamlu.mica.ai.example.pipeline.enhance.EnhanceUpscalePipeline;
import net.dreamlu.mica.ai.example.pipeline.enhance.EnhanceUpscaleResult;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * 图像超分流水线集成测试：加载真实 ONNX 模型跑完整链路。
 *
 * <p>模型缺失时整类跳过，不阻塞新克隆仓库的构建。
 */
class EnhanceUpscaleIntegrationTest {

	private static EnhanceEngine seedEngine;
	private static EnhanceEnginePool enginePool;
	private static EnhanceUpscalePipeline pipeline;

	@BeforeAll
	static void setUp() {
		nu.pattern.OpenCV.loadLocally();

		Path model = repoFile("model-tools/enhance/models/realesr-general-x4v3.onnx");
		if (model == null) {
			return;
		}
		seedEngine = EnhanceEngine.create(EnhanceConfig.builder()
			.modelPath(model.toAbsolutePath().toString()).build());
		enginePool = new EnhanceEnginePool(seedEngine);
		pipeline = new EnhanceUpscalePipeline(enginePool);
	}

	@AfterAll
	static void tearDown() {
		if (enginePool != null) {
			enginePool.destroy();
		}
		if (seedEngine != null) {
			seedEngine.close();
		}
	}

	@Test
	@DisplayName("输出倍率 4×：结果尺寸应为输入的 4 倍")
	void fourTimesScaleShouldMultiplySize() {
		assumeReady();
		EnhanceUpscaleResult result = pipeline.upscale(syntheticImage(300, 200),
			EnhanceUpscaleOptions.builder().outScale(4).tileSize(256).build());

		assertThat(result.getSourceWidth()).isEqualTo(300);
		assertThat(result.getSourceHeight()).isEqualTo(200);
		// 对比用的原图与送进模型的是同一张，尺寸必须一致
		assertThat(result.getOriginalWidth()).isEqualTo(300);
		assertThat(result.getOriginalHeight()).isEqualTo(200);
		assertThat(result.getUpscaledWidth()).isEqualTo(1200);
		assertThat(result.getUpscaledHeight()).isEqualTo(800);
		assertThat(result.isPreDownscaled()).isFalse();
		assertThat(result.getOutScale()).isEqualTo(4);
		assertThat(result.getModelScale()).isEqualTo(4);

		// 解码回来看真实像素，别只信上报的尺寸
		Mat decoded = EnhanceImageUtils.byteArrayToMat(result.getUpscaledPng());
		try {
			assertThat(decoded.cols()).isEqualTo(1200);
			assertThat(decoded.rows()).isEqualTo(800);
		} finally {
			decoded.release();
		}
	}

	@Test
	@DisplayName("输出倍率 2×：先按 4× 推理再降采样到目标尺寸")
	void twoTimesScaleShouldResampleToTarget() {
		assumeReady();
		EnhanceUpscaleResult result = pipeline.upscale(syntheticImage(300, 200),
			EnhanceUpscaleOptions.builder().outScale(2).tileSize(256).build());

		assertThat(result.getUpscaledWidth()).isEqualTo(600);
		assertThat(result.getUpscaledHeight()).isEqualTo(400);
		assertThat(result.getOutScale()).isEqualTo(2);
	}

	@Test
	@DisplayName("输入长边超上限：先等比缩小再超分，并如实上报原始尺寸")
	void oversizedInputShouldBePreDownscaled() {
		assumeReady();
		EnhanceUpscaleResult result = pipeline.upscale(syntheticImage(300, 200),
			EnhanceUpscaleOptions.builder().outScale(4).tileSize(256).maxInputSide(150).build());

		assertThat(result.isPreDownscaled()).isTrue();
		// 上传原尺寸保持不变
		assertThat(result.getSourceWidth()).isEqualTo(300);
		assertThat(result.getSourceHeight()).isEqualTo(200);
		// 送进模型的是缩小后的
		assertThat(result.getOriginalWidth()).isEqualTo(150);
		assertThat(result.getOriginalHeight()).isEqualTo(100);
		assertThat(result.getUpscaledWidth()).isEqualTo(600);
		assertThat(result.getUpscaledHeight()).isEqualTo(400);
	}

	@Test
	@DisplayName("处理档位（分块大小）只影响速度与内存，不影响输出尺寸")
	void tileTierShouldNotChangeOutputSize() {
		assumeReady();
		byte[] source = syntheticImage(300, 200);
		EnhanceUpscaleResult smallTile = pipeline.upscale(source,
			EnhanceUpscaleOptions.builder().outScale(4).tileSize(128).build());
		EnhanceUpscaleResult bigTile = pipeline.upscale(source,
			EnhanceUpscaleOptions.builder().outScale(4).tileSize(256).build());

		assertThat(smallTile.getUpscaledWidth()).isEqualTo(bigTile.getUpscaledWidth());
		assertThat(smallTile.getUpscaledHeight()).isEqualTo(bigTile.getUpscaledHeight());
		assertThat(smallTile.getTileSize()).isEqualTo(128);
		assertThat(bigTile.getTileSize()).isEqualTo(256);
	}

	@Test
	@DisplayName("引擎池：同档位复用同一引擎，不同档位各自独立")
	void enginePoolShouldReuseEnginePerTileSize() {
		assumeReady();
		assertThat(enginePool.engine(256)).isSameAs(enginePool.engine(256));
		assertThat(enginePool.engine(128)).isNotSameAs(enginePool.engine(256));
		assertThat(enginePool.modelScale()).isEqualTo(4);
	}

	private static void assumeReady() {
		assumeTrue(pipeline != null, "Real-ESRGAN 模型缺失，跳过图像超分集成测试");
	}

	/**
	 * 造一张带渐变的测试图（比纯色更能反映超分细节）。
	 */
	private static byte[] syntheticImage(int width, int height) {
		Mat mat = new Mat(height, width, CvType.CV_8UC3);
		try {
			byte[] data = new byte[width * height * 3];
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					int i = (y * width + x) * 3;
					data[i] = (byte) (x * 255 / width);
					data[i + 1] = (byte) (y * 255 / height);
					data[i + 2] = (byte) ((x + y) * 255 / (width + height));
				}
			}
			mat.put(0, 0, data);
			return EnhanceImageUtils.imencodePng(mat);
		} finally {
			mat.release();
		}
	}

	/**
	 * 从当前工作目录向上查找仓库根。
	 */
	private static Path repoFile(String relative) {
		Path dir = Paths.get("").toAbsolutePath();
		for (int i = 0; i < 6 && dir != null; i++) {
			Path candidate = dir.resolve(relative);
			if (Files.exists(candidate)) {
				return candidate;
			}
			dir = dir.getParent();
		}
		return null;
	}

}
