package net.dreamlu.mica.ai.ppocr.preproc;

import lombok.ToString;
import net.dreamlu.mica.ai.ppocr.util.NpUtil;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.List;

/**
 * 识别模型预处理：OCRReisizeNormImg + ToBatch。
 *
 * <p>对应 Python 端的 {@code RecPreProcess}：
 * <ul>
 *   <li>固定高度 48，宽度按宽高比动态计算（最小 320，最大 3200）</li>
 *   <li>归一化到 {@code [-1, 1]}（{@code (x/255 - 0.5) / 0.5}）</li>
 *   <li>批内右侧零填充对齐到同一宽度</li>
 * </ul>
 *
 * <p>输出张量用 flat {@code float[]}（NCHW 顺序，去掉前导 N 维的 N 重复），
 * 供 {@code OnnxTensor.createTensor(env, buffer, new long[]{N, 3, 48, W})} 消费。
 */
@ToString
public final class RecPreProcess {
    /**
     * 通道数（固定 3）。
     */
    public static final int CHANNELS = 3;
    /**
     * 归一化高度（固定 48）。
     */
    public static final int HEIGHT = 48;
    /**
     * 单图最小宽度（默认 320）。
     */
    public static final int W_MIN = 320;
    /**
     * 单图最大宽度（默认 3200）。
     */
    public static final int W_MAX = 3200;
    /**
     * 识别张量形状：(C, H, W_min)。
     */
    public static final int[] DEFAULT_SHAPE = {CHANNELS, HEIGHT, W_MIN};

    private final int h;
    private final int wMin;
    private final int wMax;

    public RecPreProcess() {
        this(HEIGHT, W_MIN, W_MAX);
    }

    public RecPreProcess(int h, int wMin, int wMax) {
        if (h <= 0) {
            throw new IllegalArgumentException("h must be > 0");
        }
        if (wMin <= 0 || wMax <= 0 || wMax < wMin) {
            throw new IllegalArgumentException("invalid wMin/wMax: " + wMin + "/" + wMax);
        }
        this.h = h;
        this.wMin = wMin;
        this.wMax = wMax;
    }

    /**
     * 批量预处理。
     *
     * @param imgs BGR 文本行图像列表（每张是单通道 8 位 Mat）
     * @return {@link Result}：NCHW float 数据与完整张量形状
     */
    public Result call(List<Mat> imgs) {
        int n = imgs.size();
        if (n == 0) {
            return new Result(new float[0], new int[]{0, CHANNELS, h, 0});
        }

        // 1) 逐图 resize + normalize + HWC→CHW
        java.util.List<float[][]> perImgChw = new java.util.ArrayList<>(n);
        int[] widths = new int[n];
        int[] actualWs = new int[n];
        for (int i = 0; i < n; i++) {
            Mat img = imgs.get(i);
            int srcH = img.rows();
            int srcW = img.cols();
            // 与 PaddleX 一致：max(W_min/H, W_src/H_src)
            double whRatio = Math.max((double) wMin / h, (double) srcW / srcH);
            int targetW = (int) (h * whRatio);

            int actualW;
            Mat resized;
            if (targetW > wMax) {
                // 截断到 wMax
                resized = new Mat();
                Imgproc.resize(img, resized, new Size(wMax, h), 0, 0, Imgproc.INTER_LINEAR);
                actualW = wMax;
                targetW = wMax;
            } else {
                actualW = Math.min(NpUtil.ceilDiv(h * srcW, srcH), targetW);
                resized = new Mat();
                Imgproc.resize(img, resized, new Size(actualW, h), 0, 0, Imgproc.INTER_LINEAR);
            }
            // 关键：每张图 padded 到 targetW（不是 actualW）。对应 Python
            // _resize_norm 中 padded = np.zeros((C, H, target_w))。
            widths[i] = targetW;
            actualWs[i] = actualW;

            // 转 float32 + 归一化到 [-1, 1]：(x/255 - 0.5) / 0.5
            // 重要：Core.divide(Mat, Scalar, Mat) 对多通道 Mat 处理有 bug
            // （会丢掉 G/R 通道并把它们写成 +Inf），所以这里直接转 float32 后
            // 用 Java 端循环做归一化。
            Mat f = NpUtil.toFloat32(resized);
            float[] hwcRaw = NpUtil.matToFlatHwc(f);
            f.release();
            int nPix = hwcRaw.length;
            for (int k = 0; k < nPix; k++) {
                hwcRaw[k] = (hwcRaw[k] / 255.0f - 0.5f) / 0.5f;
            }
            float[] hwcFlat = hwcRaw;
            float[][] chw = new float[CHANNELS][h * actualW];
            int hw = h * actualW;
            for (int j = 0; j < hw; j++) {
                int baseHwc = j * CHANNELS;
                chw[0][j] = hwcFlat[baseHwc];
                chw[1][j] = hwcFlat[baseHwc + 1];
                chw[2][j] = hwcFlat[baseHwc + 2];
            }
            perImgChw.add(chw);
        }

        // 2) 批内右零填充到 maxW
        int maxW = 0;
        for (int w : widths) {
            if (w > maxW) maxW = w;
        }
        int totalSize = n * CHANNELS * h * maxW;
        float[] data = new float[totalSize];
        int chwSize = CHANNELS * h * maxW;
        for (int i = 0; i < n; i++) {
            int actualW = actualWs[i];
            int destBase = i * chwSize;
            float[][] chw = perImgChw.get(i);
            // NCHW 布局：chw[c][j] = (C=c, H=j/actualW, W=j%actualW) 处的值。
            // 需要将 chw[c][j] 写到 data[c*h*maxW + (j/actualW)*maxW + (j%actualW)]。
            // 修复前直接 arraycopy 会错位（H=1, W=0 会被写到 (H=0, W=actualW)），
            // 在 maxW=actualW 时无问题，但 maxW>actualW 后 padding 部分被错误填充。
            for (int c = 0; c < CHANNELS; c++) {
                float[] chwC = chw[c];
                int cOffset = destBase + c * h * maxW;
                int hw = h * actualW;
                for (int j = 0; j < hw; j++) {
                    int hh = j / actualW;
                    int ww = j % actualW;
                    data[cOffset + hh * maxW + ww] = chwC[j];
                }
            }
        }

        return new Result(data, new int[]{n, CHANNELS, h, maxW});
    }

    /**
     * 批量预处理结果。
     */
    public record Result(float[] data, int[] shape) {
    }
}
