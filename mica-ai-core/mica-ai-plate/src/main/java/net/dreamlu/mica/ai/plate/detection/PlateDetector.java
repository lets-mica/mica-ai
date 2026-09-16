/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.plate.detection;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.plate.PlateConfig;
import net.dreamlu.mica.ai.plate.util.PlateImageUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * 车牌检测器（HyperLPR3 yolov5 多任务检测，y5fu_320x / y5fu_640x）。
 *
 * <p>输出 15 维：{@code [x_center, y_center, w, h, obj_conf, 4 关键点(8 维), 2 类score]}。
 * 后处理做 xywh2xyxy + conf*class + NMS + 坐标反算（letter_box 还原）。
 */
@Slf4j
public class PlateDetector implements AutoCloseable {

    private static final int FEATURE_DIM = 15;

    private final OrtEnvironment environment;
    private final OnnxModelSession session;
    @Getter
    private final int inputSize;
    private final float nmsThreshold;
    private final float confThreshold;

    public PlateDetector(OrtEnvironment environment, PlateConfig config,
                         OrtSession.SessionOptions sessionOptions) {
        this.environment = environment;
        this.inputSize = config.getDetectionInputSize();
        this.nmsThreshold = config.getDetectionNmsThreshold();
        this.confThreshold = config.getDetectionConfidenceThreshold();
        this.session = new OnnxModelSession(
            environment, config.getDetectionModelPath(),
            sessionOptions, "plate-detector");
    }

    public List<Detection> detect(Mat bgr) {
        try {
            LetterBoxResult pre = letterBox(bgr, inputSize);
            float[] tensorData = pre.tensorData;
            long[] shape = new long[]{1, 3, inputSize, inputSize};

            try (OnnxTensor tensor = OnnxTensor.createTensor(
                    environment, FloatBuffer.wrap(tensorData), shape);
                 OrtSession.Result result = session.getSession().run(
                    Collections.singletonMap(session.getSession().getInputNames().iterator().next(), tensor))) {

                Object outVal = result.get(0).getValue();
                float[][][] raw = toThreeDim(outVal);
                if (raw == null) {
                    return Collections.emptyList();
                }
                return postProcess(raw[0], pre.ratio, pre.left, pre.top,
                    inputSize, inputSize, confThreshold, nmsThreshold);
            } catch (OrtException e) {
                throw new MicaAiException(
                    ErrorCode.INFERENCE_FAILED,
                    "车牌检测推理失败", e);
            }
        } catch (Exception e) {
            if (e instanceof MicaAiException) {
                throw (MicaAiException) e;
            }
            throw new MicaAiException(
                ErrorCode.INFERENCE_FAILED, "车牌检测异常", e);
        }
    }

    static LetterBoxResult letterBox(Mat bgr, int size) {
        int h = bgr.rows();
        int w = bgr.cols();
        double r = Math.min(size / (double) h, size / (double) w);
        int newH = (int) Math.round(h * r);
        int newW = (int) Math.round(w * r);
        int top = (size - newH) / 2;
        int left = (size - newW) / 2;

        Mat resized = new Mat();
        Mat padded = new Mat();
        Mat rgb = new Mat();
        try {
            Imgproc.resize(bgr, resized, new Size(newW, newH));
            Core.copyMakeBorder(resized, padded, top, size - newH - top,
                left, size - newW - left, Core.BORDER_CONSTANT, new Scalar(0, 0, 0));
            Imgproc.cvtColor(padded, rgb, Imgproc.COLOR_BGR2RGB);
            float[] data = PlateImageUtils.bgrHwcToRgbChwFloat(rgb, 1.0f / 255f, 0f);
            return new LetterBoxResult(data, r, left, top);
        } finally {
            resized.release();
            padded.release();
            rgb.release();
        }
    }

    static List<Detection> postProcess(float[][] predictions,
                                        double r, int left, int top,
                                        int inputW, int inputH,
                                        float confThreshold, float nmsThreshold) {
        List<Detection> out = new ArrayList<>();
        for (float[] det : predictions) {
            if (det.length < FEATURE_DIM) {
                continue;
            }
            float objConf = det[4];
            if (objConf <= confThreshold) {
                continue;
            }
            float classScore1 = det[13] * objConf;
            float classScore2 = det[14] * objConf;
            float maxClassScore = Math.max(classScore1, classScore2);
            if (maxClassScore < confThreshold) {
                continue;
            }
            int layerNum = classScore1 >= classScore2 ? 0 : 1;

            float cx = det[0];
            float cy = det[1];
            float w = det[2];
            float h = det[3];
            float x1 = cx - w / 2f;
            float y1 = cy - h / 2f;
            float x2 = cx + w / 2f;
            float y2 = cy + h / 2f;

            x1 = (x1 - left) / (float) r;
            y1 = (y1 - top) / (float) r;
            x2 = (x2 - left) / (float) r;
            y2 = (y2 - top) / (float) r;

            int[][] landmarks = new int[4][2];
            int lmIndex = 5;
            for (int i = 0; i < 4; i++) {
                float lx = (det[lmIndex++] - left) / (float) r;
                float ly = (det[lmIndex++] - top) / (float) r;
                landmarks[i][0] = Math.max(0, Math.round(lx));
                landmarks[i][1] = Math.max(0, Math.round(ly));
            }

            Detection d = new Detection();
            d.boundingBox = new int[]{
                Math.max(0, Math.round(x1)),
                Math.max(0, Math.round(y1)),
                Math.max(0, Math.round(x2)),
                Math.max(0, Math.round(y2))
            };
            d.landmarks = landmarks;
            d.detectionScore = maxClassScore;
            d.layerNum = layerNum;
            out.add(d);
        }
        return nms(out, nmsThreshold);
    }

    static List<Detection> nms(List<Detection> boxes, float threshold) {
        boxes.sort((a, b) -> Float.compare(b.detectionScore, a.detectionScore));
        List<Detection> kept = new ArrayList<>();
        for (Detection d : boxes) {
            boolean suppressed = false;
            for (Detection k : kept) {
                if (d.layerNum != k.layerNum) {
                    continue;
                }
                if (iou(d.boundingBox, k.boundingBox) > threshold) {
                    suppressed = true;
                    break;
                }
            }
            if (!suppressed) {
                kept.add(d);
            }
        }
        return kept;
    }

    static float iou(int[] a, int[] b) {
        int x1 = Math.max(a[0], b[0]);
        int y1 = Math.max(a[1], b[1]);
        int x2 = Math.min(a[2], b[2]);
        int y2 = Math.min(a[3], b[3]);
        int inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        if (inter <= 0) {
            return 0f;
        }
        float areaA = (a[2] - a[0]) * (a[3] - a[1]);
        float areaB = (b[2] - b[0]) * (b[3] - b[1]);
        return inter / (areaA + areaB - inter);
    }

    static float[][][] toThreeDim(Object value) {
        if (value instanceof float[][][]) {
            return (float[][][]) value;
        }
        if (value instanceof float[][]) {
            float[][] two = (float[][]) value;
            return new float[][][]{two};
        }
        if (value instanceof float[]) {
            float[] flat = (float[]) value;
            return new float[][][]{new float[][]{flat}};
        }
        return null;
    }

    @Override
    public void close() {
        if (session != null) {
            session.close();
        }
    }

    public static class Detection {
        public int[] boundingBox;
        public int[][] landmarks;
        public float detectionScore;
        public int layerNum;
    }

    static class LetterBoxResult {
        final float[] tensorData;
        final double ratio;
        final int left;
        final int top;

        LetterBoxResult(float[] tensorData, double ratio, int left, int top) {
            this.tensorData = tensorData;
            this.ratio = ratio;
            this.left = left;
            this.top = top;
        }
    }
}