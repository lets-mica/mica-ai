/*
 * Copyright (c) 2024-2026 mica-ai
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
import net.dreamlu.mica.ai.plate.PlateConfig;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.Collections;

/**
 * 车牌类型分类器（HyperLPR3 litemodel_cls_96x，96×96 → 3 类 logits）。
 *
 * <p>仅在 Pipeline 根据首字符不能确定车牌颜色时调用，对应 Python 版
 * {@code self.classifier(pad)}。
 */
@Slf4j
public class PlateClassifier implements AutoCloseable {

    public static final int YELLOW = 0;
    public static final int BLUE = 1;
    public static final int GREEN = 2;

    private final OrtEnvironment environment;
    private final OnnxModelSession session;
    @Getter
    private final int inputSize;

    public PlateClassifier(OrtEnvironment environment, PlateConfig config,
                           OrtSession.SessionOptions sessionOptions) {
        this.environment = environment;
        this.inputSize = config.getClassificationInputSize();
        this.session = new OnnxModelSession(
            environment, config.getClassificationModelPath(),
            sessionOptions, "plate-classifier");
    }

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