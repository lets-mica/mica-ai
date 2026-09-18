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
package net.dreamlu.mica.ai.plate.alignment;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

/**
 * 车牌四点透视校正，对齐 HyperLPR3 Python 版 {@code get_rotate_crop_image}。
 *
 * <p>输入 4 个角点 (x,y)，输出按最大边距比例校正后的车牌图；若高宽比 ≥ 1.5 再旋转 90°
 * （双层车牌场景）。
 */
public final class PlateAligner {

    private PlateAligner() {
    }

    /**
     * 透视校正 + 必要旋转。
     *
     * @param srcBgr 输入 BGR 图
     * @param landmarks 4×2 角点，按 左上 / 右上 / 右下 / 左下 顺序（或任意四边形）
     * @return 校正后的车牌 BGR 图（Mat，调用方负责 release）
     */
    public static Mat rotateCrop(Mat srcBgr, int[][] landmarks) {
        if (landmarks == null || landmarks.length != 4) {
            throw new IllegalArgumentException("landmarks must be 4 points");
        }
        double widthTop = distance(landmarks[0], landmarks[1]);
        double widthBottom = distance(landmarks[2], landmarks[3]);
        double heightLeft = distance(landmarks[0], landmarks[3]);
        double heightRight = distance(landmarks[1], landmarks[2]);

        int cropWidth = (int) Math.max(widthTop, widthBottom);
        int cropHeight = (int) Math.max(heightLeft, heightRight);

        Mat src = new Mat(4, 2, CvType.CV_32FC1);
        Mat dst = new Mat(4, 2, CvType.CV_32FC1);
        try {
            for (int i = 0; i < 4; i++) {
                src.put(i, 0, landmarks[i][0]);
                src.put(i, 1, landmarks[i][1]);
            }
            dst.put(0, 0, 0, 0);
            dst.put(1, 0, cropWidth, 0);
            dst.put(2, 0, cropWidth, cropHeight);
            dst.put(3, 0, 0, cropHeight);

            Mat transform = Imgproc.getPerspectiveTransform(src, dst);
            Mat warped = new Mat();
            Imgproc.warpPerspective(srcBgr, warped, transform,
                new org.opencv.core.Size(cropWidth, cropHeight),
                Imgproc.INTER_CUBIC,
                Core.BORDER_REPLICATE,
                new org.opencv.core.Scalar(0, 0, 0));

            double h = warped.rows();
            double w = warped.cols();
            if (h * 1.0 / w >= 1.5) {
                Core.rotate(warped, warped, Core.ROTATE_90_COUNTERCLOCKWISE);
            }
            transform.release();
            return warped;
        } finally {
            src.release();
            dst.release();
        }
    }

    private static double distance(int[] a, int[] b) {
        double dx = a[0] - b[0];
        double dy = a[1] - b[1];
        return Math.sqrt(dx * dx + dy * dy);
    }
}