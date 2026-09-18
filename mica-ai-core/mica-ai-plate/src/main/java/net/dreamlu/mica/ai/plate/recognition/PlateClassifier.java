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
package net.dreamlu.mica.ai.plate.recognition;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.plate.util.PlateImageUtils;
import net.dreamlu.mica.ai.plate.config.PlateConfig;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.Collections;

/**
 * 车牌类型分类器（HyperLPR3 {@code litemodel_cls_96x}，96×96 → 3 类 logits）。
 *
 * <p>仅在 Pipeline 根据首字符不能确定车牌颜色时调用，对应 Python 版
 * {@code self.classifier(pad)}。
 *
 * <p>线程安全：内部 ONNX session 线程安全；通常由 {@link net.dreamlu.mica.ai.plate.pipeline.PlatePipeline}
 * 单例持有。
 */
@Slf4j
public class PlateClassifier implements AutoCloseable {

    /** 车牌颜色类型：黄色 */
    public static final int YELLOW = 0;
    /** 车牌颜色类型：蓝色 */
    public static final int BLUE = 1;
    /** 车牌颜色类型：绿色 */
    public static final int GREEN = 2;

    private final OrtEnvironment environment;
    private final OnnxModelSession session;
    @Getter
    private final int inputSize;

    /**
     * 构造车牌类型分类器。
     *
     * @param environment    ONNX 运行环境
     * @param config         车牌识别配置
     * @param sessionOptions ONNX session 选项
     */
    public PlateClassifier(OrtEnvironment environment, PlateConfig config,
                           OrtSession.SessionOptions sessionOptions) {
        this.environment = environment;
        this.inputSize = config.getClassificationInputSize();
        this.session = new OnnxModelSession(
            environment, config.getClassificationModelPath(),
            sessionOptions, "plate-classifier");
    }

    /**
     * 分类车牌颜色类型。
     *
     * @param bgr BGR 格式车牌图像
     * @return 车牌颜色类型索引：0 黄色、1 蓝色、2 绿色
     * @throws MicaAiException 推理失败时抛出，{@link ErrorCode#INFERENCE_FAILED}
     */
    public int classify(Mat bgr) {
        try {
            Mat resized = new Mat();
            try {
                Imgproc.resize(bgr, resized, new Size(inputSize, inputSize));
                float[] data = PlateImageUtils.bgrHwcToRgbChwFloat(resized, 1.0f / 255f, 0f);
                long[] shape = new long[]{1, 3, inputSize, inputSize};

                try (OnnxTensor tensor = OnnxTensor.createTensor(
                        environment, FloatBuffer.wrap(data), shape);
                     OrtSession.Result result = session.getSession().run(
                        Collections.singletonMap(session.getSession().getInputNames().iterator().next(), tensor))) {

                    float[] out = toFloatArray(result.get(0).getValue());
                    int best = 0;
                    float bestVal = Float.NEGATIVE_INFINITY;
                    for (int i = 0; i < out.length; i++) {
                        if (out[i] > bestVal) {
                            bestVal = out[i];
                            best = i;
                        }
                    }
                    return best;
                } catch (OrtException e) {
                    throw new MicaAiException(
                        ErrorCode.INFERENCE_FAILED,
                        "车牌分类推理失败", e);
                }
            } finally {
                resized.release();
            }
        } catch (Exception e) {
            if (e instanceof MicaAiException) {
                throw (MicaAiException) e;
            }
            throw new MicaAiException(
                ErrorCode.INFERENCE_FAILED, "车牌分类异常", e);
        }
    }

    static float[] toFloatArray(Object value) {
        if (value instanceof float[]) {
            return (float[]) value;
        }
        if (value instanceof float[][]) {
            float[][] two = (float[][]) value;
            return two.length > 0 ? two[0] : new float[0];
        }
        return new float[0];
    }

    @Override
    public void close() {
        if (session != null) {
            session.close();
        }
    }
}
