/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.plate.util;

import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * 车牌能力内部图像工具（零 Spring、零 face 依赖）。
 */
public final class PlateImageUtils {

    private PlateImageUtils() {
    }

    public static Mat byteArrayToMat(byte[] imageBytes) {
        if (imageBytes == null || imageBytes.length == 0) {
            throw new MicaAiException(
                ErrorCode.DETECTION_FAILED, "图像字节为空");
        }
        MatOfByte mob = new MatOfByte(imageBytes);
        try {
            Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
            if (mat.empty()) {
                throw new MicaAiException(
                    ErrorCode.DETECTION_FAILED, "图像解码失败或格式不支持");
            }
            return mat;
        } finally {
            mob.release();
        }
    }

    public static float[] bgrHwcToRgbChwFloat(Mat bgr, float scale, float offset) {
        int h = bgr.rows();
        int w = bgr.cols();
        int hw = h * w;
        byte[] pixels = new byte[hw * 3];
        bgr.get(0, 0, pixels);
        float[] data = new float[3 * hw];
        for (int i = 0; i < hw; i++) {
            int b = pixels[i * 3] & 0xFF;
            int g = pixels[i * 3 + 1] & 0xFF;
            int r = pixels[i * 3 + 2] & 0xFF;
            data[i] = (r - offset) * scale;
            data[hw + i] = (g - offset) * scale;
            data[2 * hw + i] = (b - offset) * scale;
        }
        return data;
    }

    public static float[] bgrHwcToChwFloat(Mat bgr, float scale, float offset) {
        int h = bgr.rows();
        int w = bgr.cols();
        int hw = h * w;
        byte[] pixels = new byte[hw * 3];
        bgr.get(0, 0, pixels);
        float[] data = new float[3 * hw];
        for (int i = 0; i < hw; i++) {
            int b = pixels[i * 3] & 0xFF;
            int g = pixels[i * 3 + 1] & 0xFF;
            int r = pixels[i * 3 + 2] & 0xFF;
            data[i] = (b - offset) * scale;
            data[hw + i] = (g - offset) * scale;
            data[2 * hw + i] = (r - offset) * scale;
        }
        return data;
    }

    public static void releaseAll(Mat... mats) {
        if (mats == null) {
            return;
        }
        for (Mat m : mats) {
            if (m != null) {
                m.release();
            }
        }
    }
}
