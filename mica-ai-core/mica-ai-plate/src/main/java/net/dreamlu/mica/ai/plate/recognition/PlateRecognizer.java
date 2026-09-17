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
import net.dreamlu.mica.ai.plate.config.PlateConfig;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 车牌字符识别（HyperLPR3 PPRCNN / CRNN + CTC）。
 *
 * <p>输入 BGR 校正后的车牌图，内部 resize 到 (recHeight × 变长 W) 再喂模型；
 * 输出 (1, 40, 6625) logits，argmax + 去重复 + 去 blank(0) 得到最终字符。
 */
@Slf4j
public class PlateRecognizer implements AutoCloseable {

    private static final int TIMESTEP = 40;
    private static final int MIN_W = 48;
    private static final int MAX_W = 160;

    private final OrtEnvironment environment;
    private final OnnxModelSession session;
    @Getter
    private final int inputHeight;
    @Getter
    private final int maxInputWidth;

    public PlateRecognizer(OrtEnvironment environment, PlateConfig config,
                           OrtSession.SessionOptions sessionOptions) {
        this.environment = environment;
        this.inputHeight = config.getRecognitionInputHeight();
        this.maxInputWidth = config.getRecognitionInputWidth();
        this.session = new OnnxModelSession(
            environment, config.getRecognitionModelPath(),
            sessionOptions, "plate-recognizer");
    }

    public RecognitionResult recognize(Mat bgr) {
        if (bgr == null || bgr.empty()) {
            return new RecognitionResult("", 0f);
        }
        try {
            float[] tensorData = preprocess(bgr);
            long[] shape = new long[]{1, 3, inputHeight, maxInputWidth};

            try (OnnxTensor tensor = OnnxTensor.createTensor(
                    environment, FloatBuffer.wrap(tensorData), shape);
                 OrtSession.Result result = session.getSession().run(
                    Collections.singletonMap(session.getSession().getInputNames().iterator().next(), tensor))) {

                Object outVal = result.get(0).getValue();
                float[][] logits = toTwoDim(outVal);
                return ctcDecode(logits);
            } catch (OrtException e) {
                throw new MicaAiException(
                    ErrorCode.INFERENCE_FAILED,
                    "车牌识别推理失败", e);
            }
        } catch (Exception e) {
            if (e instanceof MicaAiException) {
                throw (MicaAiException) e;
            }
            throw new MicaAiException(
                ErrorCode.INFERENCE_FAILED, "车牌识别异常", e);
        }
    }

    float[] preprocess(Mat bgr) {
        int h = bgr.rows();
        int w = bgr.cols();
        double ratio = w / (double) h;
        int targetW = Math.max(MIN_W, Math.min(MAX_W,
            (int) Math.ceil(inputHeight * ratio)));
        targetW = Math.max(MIN_W, targetW);

        Mat resized = new Mat();
        try {
            Imgproc.resize(bgr, resized, new Size(targetW, inputHeight));
            float[] data = PlateImageUtils.bgrHwcToChwFloat(resized, 1.0f / 127.5f, 127.5f);
            float[] padded = new float[3 * inputHeight * maxInputWidth];
            int srcStride = inputHeight * targetW;
            for (int c = 0; c < 3; c++) {
                for (int row = 0; row < inputHeight; row++) {
                    System.arraycopy(data, c * srcStride + row * targetW,
                        padded, c * inputHeight * maxInputWidth + row * maxInputWidth,
                        targetW);
                }
            }
            return padded;
        } finally {
            resized.release();
        }
    }

    static float[][] toTwoDim(Object value) {
        if (value instanceof float[][]) {
            return (float[][]) value;
        }
        if (value instanceof float[][][]) {
            float[][][] three = (float[][][]) value;
            return three[0];
        }
        if (value instanceof float[]) {
            float[] one = (float[]) value;
            return new float[][]{one};
        }
        return new float[0][];
    }

    static RecognitionResult ctcDecode(float[][] logits) {
        if (logits.length == 0) {
            return new RecognitionResult("", 0f);
        }
        int t = Math.min(logits.length, TIMESTEP);
        int vocab = logits[0].length;
        if (vocab != PlateDictionary.size()) {
            vocab = Math.min(vocab, PlateDictionary.size());
        }
        List<Character> chars = new ArrayList<>();
        List<Float> confs = new ArrayList<>();
        int prev = -1;
        for (int i = 0; i < t; i++) {
            int best = 0;
            float bestVal = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < vocab; j++) {
                if (logits[i][j] > bestVal) {
                    bestVal = logits[i][j];
                    best = j;
                }
            }
            if (best == PlateDictionary.BLANK_INDEX) {
                prev = -1;
                continue;
            }
            if (best == prev) {
                continue;
            }
            prev = best;
            if (best >= 0 && best < PlateDictionary.size()) {
                chars.add(PlateDictionary.TOKENS.get(best).charAt(0));
                confs.add(bestVal);
            }
        }
        if (chars.isEmpty()) {
            return new RecognitionResult("", 0f);
        }
        StringBuilder sb = new StringBuilder();
        for (Character c : chars) {
            sb.append(c);
        }
        float avgConf = 0f;
        for (float v : confs) {
            avgConf += v;
        }
        avgConf /= confs.size();
        return new RecognitionResult(sb.toString(), avgConf);
    }

    @Override
    public void close() {
        if (session != null) {
            session.close();
        }
    }

    public static class RecognitionResult {
        public final String text;
        public final float confidence;

        public RecognitionResult(String text, float confidence) {
            this.text = text;
            this.confidence = confidence;
        }
    }
}
