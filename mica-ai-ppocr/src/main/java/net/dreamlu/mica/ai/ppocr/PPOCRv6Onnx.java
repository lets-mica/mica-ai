package net.dreamlu.mica.ai.ppocr;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.ppocr.postproc.CTCLabelDecode;
import net.dreamlu.mica.ai.ppocr.postproc.DBPostProcess;
import net.dreamlu.mica.ai.ppocr.preproc.DetPreProcess;
import net.dreamlu.mica.ai.ppocr.preproc.RecPreProcess;
import net.dreamlu.mica.ai.ppocr.util.BoxUtil;
import net.dreamlu.mica.ai.ppocr.util.CropUtil;
import net.dreamlu.mica.ai.ppocr.util.NpUtil;
import org.opencv.core.Mat;

import java.io.Closeable;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * PP-OCRv6 纯 ONNX Runtime 推理器。
 *
 * <p>对应 Python 端的 {@code PPOCRv6Onnx}。默认使用 CPU provider 以保证
 * 跨平台 bit-exact 精度；如需 GPU / CoreML 加速，请传递
 * {@code preferAccelerator=true}。
 *
 * <p>典型用法：
 * <pre>{@code
 * try (PPOCRv6Onnx ocr = new PPOCRv6Onnx(
 *         "models/PP-OCRv6_tiny_det_onnx/inference.onnx",
 *         "models/PP-OCRv6_tiny_rec_0515_onnx/inference.onnx",
 *         "models/rec_char_dict.txt")) {
 *     Mat img = Imgcodecs.imread("test.png");
 *     List<OCRResult> results = ocr.run(img);
 *     for (OCRResult r : results) {
 *         System.out.println(r.text() + "  " + r.score());
 *     }
 * }
 * }</pre>
 */
@Slf4j
public final class PPOCRv6Onnx implements Closeable, AutoCloseable {
    private final OrtEnvironment env;
    private final OrtSession detSession;
    private final OrtSession recSession;
    private final String detInputName;
    private final String recInputName;

    private final DetPreProcess detPre;
    private final DBPostProcess detPost;
    private final RecPreProcess recPre;
    private final CTCLabelDecode recPost;
    private final int recBatchSize;

    private boolean closed = false;

    public PPOCRv6Onnx(String detModelPath, String recModelPath, String recCharDictPath) {
        this(detModelPath, recModelPath, recCharDictPath, Config.defaults());
    }

    public PPOCRv6Onnx(String detModelPath, String recModelPath, String recCharDictPath,
                       Config config) {
        requireFile(detModelPath, "detModelPath");
        requireFile(recModelPath, "recModelPath");
        requireFile(recCharDictPath, "recCharDictPath");
        if (config.recBatchSize < 1) {
            throw new IllegalArgumentException("recBatchSize must be >= 1, got " + config.recBatchSize);
        }
        if (config.recImageShape == null || config.recImageShape.length != 3) {
            throw new IllegalArgumentException("recImageShape must be [C, H, W]");
        }
        String[] providers = OrtProviders.resolve(!config.preferAccelerator);
        this.env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        // 默认 CPU 优化
        try {
            opts.setIntraOpNumThreads(Math.max(1, config.intraOpNumThreads));
            opts.setInterOpNumThreads(Math.max(1, config.interOpNumThreads));
        } catch (OrtException e) {
            log.warn("设置线程数失败，使用默认值: {}", e.getMessage());
        }

        try {
            this.detSession = env.createSession(detModelPath, opts);
            this.recSession = env.createSession(recModelPath, opts);
        } catch (OrtException e) {
            throw new RuntimeException("创建 ONNX Runtime 会话失败: " + e.getMessage(), e);
        }

        this.detInputName = detSession.getInputNames().iterator().next();
        this.recInputName = recSession.getInputNames().iterator().next();

        this.detPre = new DetPreProcess(config.detLimitSideLen, config.detLimitType, config.detMaxSideLimit);
        this.detPost = new DBPostProcess(config.detThresh, config.detBoxThresh, config.detUnclipRatio,
                1000, 3);
        this.recPre = new RecPreProcess(config.recImageShape[1], 320, 3200);
        this.recPost = new CTCLabelDecode(recCharDictPath);
        this.recBatchSize = config.recBatchSize;

        log.info("PPOCRv6Onnx 初始化完成: det={}, rec={}, vocab={}",
                this.detPre, this.recPre, this.recPost.vocabSize());
    }

    @Override
    public void close() {
        if (!closed) {
            try {
                if (detSession != null) detSession.close();
            } catch (OrtException e) {
                log.debug("关闭 det session 失败: {}", e.getMessage());
            }
            try {
                if (recSession != null) recSession.close();
            } catch (OrtException e) {
                log.debug("关闭 rec session 失败: {}", e.getMessage());
            }
            closed = true;
            log.info("PPOCRv6Onnx 已关闭");
        }
    }

    private void requireOpen() {
        if (closed) {
            throw new IllegalStateException("PPOCRv6Onnx has been closed and can no longer be used.");
        }
    }

    @Override
    public String toString() {
        return "PPOCRv6Onnx(det=" + detPre + ", rec=" + recPre
                + ", vocab=" + recPost.vocabSize() + ", closed=" + closed + ")";
    }

    // ==================================================================
    // 检测
    // ==================================================================

    /**
     * 文本检测。
     *
     * @param imgBgr BGR 格式图像 (H, W, 3) uint8
     * @return {@link DetectResult} boxes 形状 (N, 4, 2) int，scores 长度 N
     */
    public DetectResult detect(Mat imgBgr) {
        requireOpen();
        DetPreProcess.Result prep = detPre.call(imgBgr);
        long[] detShape = toLongArray(prep.shape());
        FloatBuffer buf = NpUtil.toBuffer(prep.data());
        Map<String, OnnxTensor> inputs = Collections.singletonMap(detInputName, tensor(buf, detShape));
        try (OrtSession.Result result = detSession.run(inputs)) {
            OnnxTensor outTensor = (OnnxTensor) result.get(0);
            float[][] prob = readProb2D(outTensor);
            DBPostProcess.Result post = detPost.call(probToMat(prob, prep.imgShape()), prep.imgShape());
            return new DetectResult(post.boxes(), post.scores());
        } catch (OrtException e) {
            throw new RuntimeException("det 推理失败: " + e.getMessage(), e);
        }
    }

    // ==================================================================
    // 识别
    // ==================================================================

    /**
     * 文本识别（支持批量）。
     *
     * @param imgList 裁剪后的 BGR 文本行图像列表
     * @return {@link RecognizeResult} texts 与 scores 长度一致
     */
    public RecognizeResult recognize(List<Mat> imgList) {
        requireOpen();
        int n = imgList.size();
        if (n == 0) {
            return new RecognizeResult(new String[0], new float[0]);
        }
        if (log.isDebugEnabled()) {
            Mat first = imgList.get(0);
            int h = first.rows(), w = first.cols(), c = first.channels();
            log.debug("rec 输入 #0: {}x{}x{} type={} (BGR)", h, w, c, first.type());
        }

        // 按宽高比排序以优化 batch padding，推理后恢复原始顺序
        List<Integer> order = new ArrayList<>(n);
        List<Double> ratios = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            Mat m = imgList.get(i);
            order.add(i);
            ratios.add((double) m.cols() / m.rows());
        }
        List<Integer> sortedOrder = new ArrayList<>(order);
        sortedOrder.sort(Comparator.comparingDouble(ratios::get));

        List<Mat> sortedImgs = new ArrayList<>(n);
        for (int idx : sortedOrder) {
            sortedImgs.add(imgList.get(idx));
        }

        String[] texts = new String[n];
        float[] scores = new float[n];

        for (int start = 0; start < n; start += recBatchSize) {
            int end = Math.min(start + recBatchSize, n);
            List<Mat> batch = sortedImgs.subList(start, end);
            RecPreProcess.Result prep = recPre.call(batch);
            long[] shape = toLongArray(prep.shape());
            FloatBuffer buf = NpUtil.toBuffer(prep.data());
            Map<String, OnnxTensor> inputs = Collections.singletonMap(recInputName, tensor(buf, shape));
            try (OrtSession.Result result = recSession.run(inputs)) {
                OnnxTensor outTensor = (OnnxTensor) result.get(0);
                float[][][] modelOutput = read3D(outTensor);
                CTCLabelDecode.Result decoded = recPost.call(modelOutput);
                for (int j = 0; j < decoded.texts().length; j++) {
                    int orig = sortedOrder.get(start + j);
                    texts[orig] = decoded.texts()[j];
                    scores[orig] = decoded.scores()[j];
                }
            } catch (OrtException e) {
                throw new RuntimeException("rec 推理失败: " + e.getMessage(), e);
            }
        }
        return new RecognizeResult(texts, scores);
    }

    // ==================================================================
    // 完整流程
    // ==================================================================

    /**
     * 完整 OCR 流程：检测 → 排序 → 裁剪 → 识别。
     *
     * @param imgBgr BGR 格式图像 (H, W, 3) uint8
     * @return OCRResult 列表（按阅读顺序排列）
     */
    public List<OCRResult> run(Mat imgBgr) {
        requireOpen();
        // 1. 检测
        DetectResult dr = detect(imgBgr);
        if (dr.boxes().length == 0) {
            return List.of();
        }

        // 2. 排序
        int[][][] sortedBoxes = BoxUtil.sortQuadBoxes(dr.boxes());

        // 3. 裁剪（失败条目为 null）
        List<Mat> crops = CropUtil.cropByPolys(imgBgr, sortedBoxes);

        // 4. 对齐过滤
        List<int[][]> validBoxes = new ArrayList<>();
        List<Mat> validCrops = new ArrayList<>();
        for (int i = 0; i < sortedBoxes.length; i++) {
            if (crops.get(i) != null) {
                validBoxes.add(sortedBoxes[i]);
                validCrops.add(crops.get(i));
            }
        }
        if (validCrops.isEmpty()) {
            return List.of();
        }

        // 5. 识别
        RecognizeResult rr = recognize(validCrops);

        List<OCRResult> results = new ArrayList<>(validBoxes.size());
        for (int i = 0; i < validBoxes.size(); i++) {
            results.add(new OCRResult(rr.texts()[i], rr.scores()[i], validBoxes.get(i)));
        }
        return results;
    }

    // ==================================================================
    // 内部：ONNX Tensor 工具
    // ==================================================================

    private OnnxTensor tensor(FloatBuffer buf, long[] shape) {
        try {
            return OnnxTensor.createTensor(env, buf, shape);
        } catch (OrtException e) {
            throw new RuntimeException("创建 OnnxTensor 失败: " + e.getMessage(), e);
        }
    }

    private long[] toLongArray(int[] arr) {
        long[] out = new long[arr.length];
        for (int i = 0; i < arr.length; i++) {
            out[i] = arr[i];
        }
        return out;
    }

    /**
     * 读取 4D (1, 1, H, W) 输出 → 2D float[H][W] 数组。
     */
    private float[][] readProb2D(OnnxTensor tensor) throws OrtException {
        FloatBuffer buf = tensor.getFloatBuffer();
        long[] shape = tensor.getInfo().getShape();
        // 期望 [1, 1, H, W]
        int total = (int) (shape[0] * shape[1] * shape[2] * shape[3]);
        float[] data = new float[total];
        buf.get(data);
        int h = (int) shape[2];
        int w = (int) shape[3];
        float[][] out = new float[h][w];
        for (int i = 0; i < h; i++) {
            System.arraycopy(data, i * w, out[i], 0, w);
        }
        return out;
    }

    /**
     * 读取 3D (B, T, C) 输出 → float[B][T][C] 数组。
     */
    private float[][][] read3D(OnnxTensor tensor) throws OrtException {
        FloatBuffer buf = tensor.getFloatBuffer();
        long[] shape = tensor.getInfo().getShape();
        if (shape.length != 3) {
            throw new IllegalArgumentException("期望 3D rec 输出, 实际 " + shape.length + "D");
        }
        int b = (int) shape[0];
        int t = (int) shape[1];
        int c = (int) shape[2];
        float[] data = new float[b * t * c];
        buf.get(data);
        float[][][] out = new float[b][t][c];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                System.arraycopy(data, (i * t + j) * c, out[i][j], 0, c);
            }
        }
        return out;
    }

    private Mat probToMat(float[][] prob, float[] imgShape) {
        int h = prob.length;
        int w = prob[0].length;
        Mat m = new Mat(h, w, org.opencv.core.CvType.CV_32F);
        float[] flat = new float[h * w];
        for (int i = 0; i < h; i++) {
            System.arraycopy(prob[i], 0, flat, i * w, w);
        }
        m.put(0, 0, flat);
        return m;
    }

    // ==================================================================
    // 静态工具
    // ==================================================================

    private static void requireFile(String path, String name) {
        if (path == null) {
            throw new IllegalArgumentException(name + " is null");
        }
        if (!Files.isRegularFile(Path.of(path))) {
            throw new IllegalArgumentException(name + ": file not found: " + path);
        }
    }

    // ==================================================================
    // 内部记录
    // ==================================================================

    public record DetectResult(int[][][] boxes, float[] scores) {
    }

    public record RecognizeResult(String[] texts, float[] scores) {
    }

    /**
     * 配置参数。
     */
    public static final class Config {
        int detLimitSideLen = 64;
        String detLimitType = "min";
        int detMaxSideLimit = 4000;
        float detThresh = 0.3f;
        float detBoxThresh = 0.6f;
        float detUnclipRatio = 1.5f;
        int[] recImageShape = {3, 48, 320};
        int recBatchSize = 6;
        boolean preferAccelerator = false;
        int intraOpNumThreads = 1;
        int interOpNumThreads = 1;

        public static Config defaults() {
            return new Config();
        }

        public Config detLimitSideLen(int v) {
            this.detLimitSideLen = v;
            return this;
        }

        public Config detLimitType(String v) {
            this.detLimitType = v;
            return this;
        }

        public Config detMaxSideLimit(int v) {
            this.detMaxSideLimit = v;
            return this;
        }

        public Config detThresh(float v) {
            this.detThresh = v;
            return this;
        }

        public Config detBoxThresh(float v) {
            this.detBoxThresh = v;
            return this;
        }

        public Config detUnclipRatio(float v) {
            this.detUnclipRatio = v;
            return this;
        }

        public Config recImageShape(int[] v) {
            this.recImageShape = v;
            return this;
        }

        public Config recBatchSize(int v) {
            this.recBatchSize = v;
            return this;
        }

        public Config preferAccelerator(boolean v) {
            this.preferAccelerator = v;
            return this;
        }

        public Config intraOpNumThreads(int v) {
            this.intraOpNumThreads = v;
            return this;
        }

        public Config interOpNumThreads(int v) {
            this.interOpNumThreads = v;
            return this;
        }
    }
}
