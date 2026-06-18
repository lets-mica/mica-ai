package net.dreamlu.mica.ai.ppocr.util;

import lombok.experimental.UtilityClass;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.nio.FloatBuffer;
import java.util.List;

/**
 * Numpy 风格工具 + 轻量 NDArray 运算。
 *
 * <p>设计原则：图像（2D Mat）走 OpenCV API，张量（多维 float 数组）走 Java
 * 原生 float[] 操作，避免 OpenCV 在 3D/4D Mat 上的 API 限制。
 * ONNX Runtime Java API 直接消费 {@link FloatBuffer}，因此 NCHW 张量
 * 用 flat float[] 表示。
 *
 * <p>实际提供的运算（仅覆盖 ppocrv6_onnx.py 中所需）：
 * <ul>
 *   <li>张量 permute / reshape：{@link #hwcFlatToNchw(float[], int, int, int)}</li>
 *   <li>沿最后一维的 {@link #argmaxLastAxis(float[][][])} / {@link #maxLastAxis(float[][][])}</li>
 *   <li>堆叠：{@link #stack3D(List)} / {@link #stack2D(List)}</li>
 *   <li>填充 / 取整 / clip</li>
 *   <li>Mat ↔ float[] 转换</li>
 * </ul>
 */
@UtilityClass
public final class NpUtil {
    // ===================================================================
    // 张量 permute / 维度变换
    // ===================================================================

    /**
     * 等价 {@code np.transpose(img, (2, 0, 1))[None, ...]}。
     *
     * <p>输入：HWC 顺序的 flat float 数组（长度 {@code H * W * C}）。
     * 输出：NCHW 顺序的 flat float 数组（长度 {@code 1 * C * H * W}），
     * 其中 N=1 直接省略（前导 1 维）。
     *
     * @param hwc  HWC 数据，length = H * W * C
     * @param h    高度
     * @param w    宽度
     * @param c    通道数（通常 3）
     * @return NCHW 数据（去掉前导 N 维），length = C * H * W
     */
    public static float[] hwcFlatToNchw(float[] hwc, int h, int w, int c) {
        float[] chw = new float[c * h * w];
        int hw = h * w;
        // hwc[i * 3 + c]  →  chw[c * hw + i]
        for (int i = 0; i < hw; i++) {
            int baseHwc = i * c;
            int baseChw = i;
            for (int ch = 0; ch < c; ch++) {
                chw[ch * hw + baseChw] = hwc[baseHwc + ch];
            }
        }
        return chw;
    }

    /**
     * 把 HWC 形状的 float Mat 读取为 flat float 数组（行优先）。
     * 假定 Mat 是连续的；若不连续先 clone。
     */
    public static float[] matToFlatHwc(Mat hwc) {
        Mat m = hwc.isContinuous() ? hwc : hwc.clone();
        int h = m.rows();
        int w = m.cols();
        int c = m.channels();
        float[] data = new float[h * w * c];
        m.get(0, 0, data);
        return data;
    }

    /**
     * Mat → FloatBuffer（张量输入到 ORT）。
     */
    public static FloatBuffer toBuffer(float[] flat) {
        return FloatBuffer.wrap(flat);
    }

    // ===================================================================
    // 沿最后一维运算
    // ===================================================================

    /**
     * 等价 {@code np.argmax(x, axis=-1)}。输入形状 {@code (B, T, C)}，输出 {@code (B, T)}。
     */
    public static int[][] argmaxLastAxis(float[][][] x) {
        int b = x.length;
        if (b == 0) {
            return new int[0][];
        }
        int t = x[0].length;
        int[][] idx = new int[b][t];
        for (int i = 0; i < b; i++) {
            float[][] ti = x[i];
            for (int j = 0; j < t; j++) {
                float[] row = ti[j];
                int bestC = 0;
                float bestV = row[0];
                for (int c = 1; c < row.length; c++) {
                    float v = row[c];
                    if (v > bestV) {
                        bestV = v;
                        bestC = c;
                    }
                }
                idx[i][j] = bestC;
            }
        }
        return idx;
    }

    /**
     * 等价 {@code np.max(x, axis=-1)}。输入形状 {@code (B, T, C)}，输出 {@code (B, T)}。
     */
    public static float[][] maxLastAxis(float[][][] x) {
        int b = x.length;
        if (b == 0) {
            return new float[0][];
        }
        int t = x[0].length;
        float[][] m = new float[b][t];
        for (int i = 0; i < b; i++) {
            float[][] ti = x[i];
            for (int j = 0; j < t; j++) {
                float[] row = ti[j];
                float bestV = row[0];
                for (int c = 1; c < row.length; c++) {
                    if (row[c] > bestV) {
                        bestV = row[c];
                    }
                }
                m[i][j] = bestV;
            }
        }
        return m;
    }

    // ===================================================================
    // 堆叠
    // ===================================================================

    /**
     * 等价 {@code np.stack(list, axis=0)} 对 {@code (4, 2)} 形状的列表。
     * 返回新分配的 {@code (N, 4, 2)} 数组，类型 float。
     */
    public static float[][][] stack3D(List<float[][]> list) {
        if (list.isEmpty()) {
            return new float[0][][];
        }
        int n = list.size();
        int r = list.get(0).length;
        int c = list.get(0)[0].length;
        float[][][] out = new float[n][r][c];
        for (int i = 0; i < n; i++) {
            float[][] src = list.get(i);
            for (int j = 0; j < r; j++) {
                System.arraycopy(src[j], 0, out[i][j], 0, c);
            }
        }
        return out;
    }

    /**
     * 等价 {@code np.stack(list, axis=0)} 对 1 维数组。
     */
    public static float[][] stack2D(List<float[]> list) {
        if (list.isEmpty()) {
            return new float[0][];
        }
        int n = list.size();
        int cols = list.get(0).length;
        float[][] out = new float[n][cols];
        for (int i = 0; i < n; i++) {
            System.arraycopy(list.get(i), 0, out[i], 0, cols);
        }
        return out;
    }

    // ===================================================================
    // 填充 / 取整 / clip
    // ===================================================================

    /**
     * 右零填充 2D 数组的列。返回新数组，行数不变，列数变为 {@code targetCols}。
     */
    public static float[][] padRight(float[][] x, int targetCols) {
        int rows = x.length;
        float[][] out = new float[rows][targetCols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(x[i], 0, out[i], 0, x[i].length);
        }
        return out;
    }

    /**
     * 整型上取整（等价 {@code math.ceil(a / b)} 但使用整数算术）。
     */
    public static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }

    /**
     * 整型 clip（{@code Math.max(min, Math.min(max, v))}）。
     */
    public static int clamp(int v, int min, int max) {
        return Math.max(min, Math.min(max, v));
    }

    /**
     * 浮点 clip。
     */
    public static float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(max, v));
    }

    /**
     * 1 维整型数组 clip。
     */
    public static int[] clipAll(int[] v, int min, int max) {
        int[] out = new int[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = clamp(v[i], min, max);
        }
        return out;
    }

    /**
     * 等价 {@code np.round(x).astype(int)}。
     */
    public static int roundToInt(float v) {
        return Math.round(v);
    }

    // ===================================================================
    // Mat 工具
    // ===================================================================

    /**
     * 等价 {@code img.astype(np.float32)}。返回新 Mat。
     */
    public static Mat toFloat32(Mat src) {
        Mat dst = new Mat();
        src.convertTo(dst, CvType.CV_32F);
        return dst;
    }

    /**
     * 等价 {@code np.empty((0, 4, 2), dtype=np.int16)}。
     */
    public static int[][][] empty3D() {
        return new int[0][][];
    }
}
