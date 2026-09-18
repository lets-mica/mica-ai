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

    /**
     * 将图像字节数组解码为 BGR 格式的 {@link Mat}。
     *
     * @param imageBytes 图像文件字节（jpg/png 等）
     * @return 解码后的 BGR 图像
     * @throws MicaAiException 字节为空或解码失败时抛出，{@link ErrorCode#DETECTION_FAILED}
     */
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

    /**
     * 将 BGR HWC 图像转为 RGB CHW float 数组。
     *
     * @param bgr    BGR 格式图像
     * @param scale  像素缩放系数
     * @param offset 像素偏移量
     * @return RGB CHW 排列的 float 数组
     */
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

    /**
     * 将 BGR HWC 图像转为 BGR CHW float 数组。
     *
     * @param bgr    BGR 格式图像
     * @param scale  像素缩放系数
     * @param offset 像素偏移量
     * @return BGR CHW 排列的 float 数组
     */
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

    /**
     * 释放所有 {@link Mat} 的本地内存，忽略 null 元素。
     *
     * @param mats 待释放的 Mat 数组
     */
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
