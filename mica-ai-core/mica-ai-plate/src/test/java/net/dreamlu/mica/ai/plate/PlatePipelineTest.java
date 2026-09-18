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
package net.dreamlu.mica.ai.plate;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.plate.config.PlateConfig;
import net.dreamlu.mica.ai.plate.model.PlateResult;
import net.dreamlu.mica.ai.plate.pipeline.PlatePipeline;
import nu.pattern.OpenCV;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@EnabledIf("isModelPresent")
class PlatePipelineTest {

    private static final Path DET_PATH = repoPath("model-tools/plate/models/y5fu_320x_sim.onnx");
    private static final Path REC_PATH = repoPath("model-tools/plate/models/rpv3_mdict_160_r3.onnx");
    private static final Path CLS_PATH = repoPath("model-tools/plate/models/litemodel_cls_96x_r1.onnx");
    private static final String TEST_IMAGE = "/images/test_img.jpg";

    private static Path repoPath(String relativePath) {
        Path dir = Paths.get("").toAbsolutePath();
        while (dir != null && !Files.exists(dir.resolve("model-tools"))) {
            dir = dir.getParent();
        }
        return dir == null ? Paths.get(relativePath) : dir.resolve(relativePath);
    }

    static boolean isModelPresent() {
        return Files.exists(DET_PATH)
            && Files.exists(REC_PATH)
            && Files.exists(CLS_PATH)
            && PlatePipelineTest.class.getResource(TEST_IMAGE) != null;
    }

    @BeforeAll
    static void loadNative() {
        OpenCV.loadLocally();
    }

    @Test
    void recognizeRealImage() {
        PlateConfig config = PlateConfig.builder()
            .detectionModelPath(DET_PATH.toString())
            .recognitionModelPath(REC_PATH.toString())
            .classificationModelPath(CLS_PATH.toString())
            .build();
        try (PlatePipeline pipeline = PlatePipeline.create(config)) {
            byte[] imageBytes = loadTestImage();
            List<PlateResult> results = pipeline.recognizeBytes(imageBytes);
            assertThat(results).isNotEmpty();
            PlateResult first = results.get(0);
            assertThat(first.getPlateCode()).isNotBlank();
            assertThat(first.getDetectionConfidence()).isGreaterThan(0.5f);
        }
    }

    private static byte[] loadTestImage() {
        try (InputStream in = PlatePipelineTest.class.getResourceAsStream(TEST_IMAGE)) {
            assertThat(in).as("test image %s on classpath", TEST_IMAGE).isNotNull();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            byte[] buffer = new byte[8192];
            int n;
            while ((n = in.read(buffer)) != -1) {
                out.write(buffer, 0, n);
            }
            return out.toByteArray();
        } catch (IOException e) {
            throw new IllegalStateException("读取测试图片失败: " + TEST_IMAGE, e);
        }
    }

    @Test
    void recognizePathMissingFileThrows() {
        PlateConfig config = PlateConfig.builder()
            .detectionModelPath(DET_PATH.toString())
            .recognitionModelPath(REC_PATH.toString())
            .classificationModelPath(CLS_PATH.toString())
            .build();
        try (PlatePipeline pipeline = PlatePipeline.create(config)) {
            assertThatThrownBy(() -> pipeline.recognizePath("non-exist.jpg"))
                .isInstanceOf(MicaAiException.class);
        }
    }
}
