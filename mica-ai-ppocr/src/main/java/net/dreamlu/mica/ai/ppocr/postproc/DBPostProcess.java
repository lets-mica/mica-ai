package net.dreamlu.mica.ai.ppocr.postproc;

import net.dreamlu.mica.ai.ppocr.util.BoxUtil;
import net.dreamlu.mica.ai.ppocr.util.NpUtil;
import net.dreamlu.mica.ai.ppocr.util.Offset;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * DB 后处理：从概率图中提取四边形文本框（boxType="quad"）。
 *
 * <p>对应 Python 端的 {@code DBPostProcess}：
 * <ol>
 *   <li>二值化（{@code prob > thresh}）</li>
 *   <li>{@code findContours} 找到候选轮廓</li>
 *   <li>对每个轮廓计算 {@code minAreaRect}，评估得分（多边形内概率均值）</li>
 *   <li>得分 &gt; {@code box_thresh} 的轮廓做 unclip（pyclipper 等价）扩展</li>
 *   <li>对扩展结果再次取 {@code minAreaRect}，映射回原图坐标</li>
 * </ol>
 */
public final class DBPostProcess {

    private static final Logger LOG = LoggerFactory.getLogger(DBPostProcess.class);

    private final float thresh;
    private final float boxThresh;
    private final float unclipRatio;
    private final int maxCandidates;
    private final int minSize;

    public DBPostProcess(float thresh, float boxThresh, float unclipRatio,
                          int maxCandidates, int minSize) {
        this.thresh = thresh;
        this.boxThresh = boxThresh;
        this.unclipRatio = unclipRatio;
        this.maxCandidates = maxCandidates;
        this.minSize = minSize;
    }

    @Override
    public String toString() {
        return "DBPostProcess(thresh=" + thresh + ", boxThresh=" + boxThresh
                + ", unclip=" + unclipRatio + ")";
    }

    /**
     * 执行 DB 后处理。
     *
     * @param prob     概率图 (H', W') 浮点；可由 ONNX Runtime 输出的 (1,1,H',W') 张量
     *                 直接 reshape 取出
     * @param imgShape {@code [srcH, srcW, ratioH, ratioW]}（det 预处理返回的 imgShape）
     * @return {@link Result}：boxes 形状 (N, 4, 2) int，scores 长度 N
     */
    public Result call(Mat prob, float[] imgShape) {
        int srcH = (int) imgShape[0];
        int srcW = (int) imgShape[1];

        if (prob.dims() == 4 && prob.size(0) == 1 && prob.size(1) == 1) {
            int hNew = (int) prob.size(2);
            int wNew = (int) prob.size(3);
            // (1,1,H,W) → (H, W) 共享 data 指针
            prob = prob.reshape(1, hNew);
            if (prob.cols() != wNew) {
                prob = prob.reshape(1, wNew);
            }
        } else if (prob.dims() != 2) {
            StringBuilder sb = new StringBuilder("DBPostProcess: 期望 prob 2D (H, W) 或 4D (1, 1, H, W)，实际 (");
            for (int d = 0; d < prob.dims(); d++) {
                if (d > 0) sb.append(", ");
                sb.append(prob.size(d));
            }
            sb.append(")");
            throw new IllegalArgumentException(sb.toString());
        }

        // 二值化（prob > thresh → 1）
        Mat segmentation = new Mat();
        Imgproc.threshold(prob, segmentation, thresh, 1.0, Imgproc.THRESH_BINARY);

        return extractBoxes(prob, segmentation, srcW, srcH);
    }

    /**
     * 从二值化位图提取文本框并映射回原始图像坐标。
     */
    private Result extractBoxes(Mat prob, Mat bitmap, int dstW, int dstH) {
        // 转换为 uint8（0/255）给 findContours
        Mat u8 = new Mat();
        Core.multiply(bitmap, new Scalar(255.0), u8);
        u8.convertTo(u8, CvType.CV_8U);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(u8, contours, hierarchy, Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_SIMPLE);

        int bmH = bitmap.rows();
        int bmW = bitmap.cols();
        double ws = (double) dstW / bmW;
        double hs = (double) dstH / bmH;

        List<int[][]> boxList = new ArrayList<>();
        List<Float> scoreList = new ArrayList<>();

        int n = Math.min(contours.size(), maxCandidates);
        for (int i = 0; i < n; i++) {
            MatOfPoint contour = contours.get(i);
            BoxUtil.MinAreaBox mab;
            try {
                mab = BoxUtil.orderMinAreaBoxPoints(contour);
            } catch (Exception e) {
                LOG.debug("minAreaRect 失败, 跳过轮廓 #{}: {}", i, e.getMessage());
                continue;
            }
            if (mab.minSideLen() < minSize) {
                continue;
            }

            float[][] pts = mab.asFloatArray();
            float score = boxScore(prob, pts);
            if (score < boxThresh) {
                continue;
            }

            // unclip
            float[][] expanded = Offset.unclip(pts, Offset.unclipDistance(pts, unclipRatio));
            if (expanded.length < 3) {
                continue;
            }

            BoxUtil.MinAreaBox mab2 = BoxUtil.orderMinAreaBoxPoints(expanded);
            if (mab2.minSideLen() < minSize + 2) {
                continue;
            }

            float[][] boxF = mab2.asFloatArray();
            int[][] boxI = new int[4][2];
            for (int k = 0; k < 4; k++) {
                int x = NpUtil.clamp((int) Math.round(boxF[k][0] * ws), 0, dstW);
                int y = NpUtil.clamp((int) Math.round(boxF[k][1] * hs), 0, dstH);
                boxI[k][0] = x;
                boxI[k][1] = y;
            }
            boxList.add(boxI);
            scoreList.add(score);
        }

        for (MatOfPoint c : contours) {
            c.release();
        }
        hierarchy.release();
        u8.release();

        int[][][] boxes = new int[boxList.size()][4][2];
        for (int i = 0; i < boxList.size(); i++) {
            boxes[i] = boxList.get(i);
        }
        float[] scores = new float[scoreList.size()];
        for (int i = 0; i < scoreList.size(); i++) {
            scores[i] = scoreList.get(i);
        }
        return new Result(boxes, scores);
    }

    /**
     * 计算多边形内的概率均值（fast 模式）。
     *
     * <p>对应 Python：{@code cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]}
     */
    private float boxScore(Mat bitmap, float[][] polygon) {
        int h = bitmap.rows();
        int w = bitmap.cols();
        float[][] box = new float[polygon.length][2];
        System.arraycopy(polygon, 0, box, 0, polygon.length);

        float xMinF = Float.POSITIVE_INFINITY, xMaxF = Float.NEGATIVE_INFINITY;
        float yMinF = Float.POSITIVE_INFINITY, yMaxF = Float.NEGATIVE_INFINITY;
        for (float[] p : box) {
            if (p[0] < xMinF) xMinF = p[0];
            if (p[0] > xMaxF) xMaxF = p[0];
            if (p[1] < yMinF) yMinF = p[1];
            if (p[1] > yMaxF) yMaxF = p[1];
        }

        int xMin = NpUtil.clamp((int) Math.floor(xMinF), 0, w - 1);
        int xMax = NpUtil.clamp((int) Math.ceil(xMaxF), 0, w - 1);
        int yMin = NpUtil.clamp((int) Math.floor(yMinF), 0, h - 1);
        int yMax = NpUtil.clamp((int) Math.ceil(yMaxF), 0, h - 1);

        if (xMax < xMin || yMax < yMin) {
            return 0f;
        }

        int ww = xMax - xMin + 1;
        int hh = yMax - yMin + 1;
        Mat mask = Mat.zeros(hh, ww, CvType.CV_8U);

        // 平移到 mask 局部坐标
        Point[] shifted = new Point[box.length];
        for (int i = 0; i < box.length; i++) {
            shifted[i] = new Point(box[i][0] - xMin, box[i][1] - yMin);
        }
        MatOfPoint mop = new MatOfPoint(shifted);
        ArrayList<MatOfPoint> list = new ArrayList<>();
        list.add(mop);
        Imgproc.fillPoly(mask, list, new Scalar(1));

        // 提取子区域概率
        Mat roi = bitmap.submat(yMin, yMax + 1, xMin, xMax + 1);

        Scalar mean = Core.mean(roi, mask);
        float score = (float) mean.val[0];
        mask.release();
        return score;
    }

    /** 后处理结果。 */
    public record Result(int[][][] boxes, float[] scores) {
    }
}
