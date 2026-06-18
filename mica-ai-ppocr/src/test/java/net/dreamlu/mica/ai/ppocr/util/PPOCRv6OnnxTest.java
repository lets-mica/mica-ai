package net.dreamlu.mica.ai.ppocr.util;

import net.dreamlu.mica.ai.ppocr.OCRResult;
import net.dreamlu.mica.ai.ppocr.PPOCRv6Onnx;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

/**
 * 命令行入口。
 *
 * <p>对应 Python 端的 {@code _main} / {@code demo.py}。
 *
 * <p>用法：
 * <pre>
 * java -jar mica-ppocr-0.1.0-all.jar [image]
 *     [--det-model PATH] [--rec-model PATH] [--dict PATH]
 *     [--save-vis PATH] [--verbose|-v]
 * </pre>
 */
public final class PPOCRv6OnnxTest {

    private static final Logger LOG = LoggerFactory.getLogger(PPOCRv6OnnxTest.class);

    private PPOCRv6OnnxTest() {
    }

    public static void main(String[] args) {
        loadOpenCV();

        Args parsed = Args.parse(args);
        if (parsed.verbose) {
            setVerboseLogging();
        }

        // 校验模型文件
        List<String> missing = new ArrayList<>();
        if (!Files.isRegularFile(Path.of(parsed.detModel))) {
            missing.add("  - det: " + parsed.detModel);
        }
        if (!Files.isRegularFile(Path.of(parsed.recModel))) {
            missing.add("  - rec: " + parsed.recModel);
        }
        if (!Files.isRegularFile(Path.of(parsed.dict))) {
            missing.add("  - dict: " + parsed.dict);
        }
        if (!missing.isEmpty()) {
            System.err.println("Error: 模型文件缺失，请先下载模型：");
            missing.forEach(System.err::println);
            System.err.println("\n下载地址参考 README.md 或原项目 README_zh.md。");
            System.exit(1);
        }

        // 选图
        Path imagePath = (parsed.image != null) ? Path.of(parsed.image) : findDemoImage();
        if (imagePath == null) {
            System.err.println("Error: 未指定图片且未在 assets/test_images 下找到任何 PNG。");
            System.exit(1);
        }
        System.out.println("Image: " + imagePath.getFileName());
        System.out.println("Det model: " + Path.of(parsed.detModel).getFileName());
        System.out.println("Rec model: " + Path.of(parsed.recModel).getFileName());

        Mat img = Imgcodecs.imread(imagePath.toAbsolutePath().toString());
        if (img == null || img.empty()) {
            System.err.println("Error: 无法读取图片 " + imagePath);
            System.exit(1);
        }
        System.out.println("Image size: " + img.cols() + "x" + img.rows());

        long t0 = System.currentTimeMillis();
        List<OCRResult> results;
        try (PPOCRv6Onnx ocr = new PPOCRv6Onnx(parsed.detModel, parsed.recModel, parsed.dict)) {
            System.out.println("Running OCR...");
            results = ocr.run(img);
        }
        long elapsed = System.currentTimeMillis() - t0;
        System.out.println("\n检测到 " + results.size() + " 个文本区域（耗时 " + elapsed + " ms）：\n");
        for (int i = 0; i < results.size(); i++) {
            OCRResult r = results.get(i);
            String boxStr = "";
            if (parsed.verbose) {
                int[][] b = r.box();
                boxStr = String.format("  box=[(%d,%d),(%d,%d),(%d,%d),(%d,%d)]",
                        b[0][0], b[0][1], b[1][0], b[1][1], b[2][0], b[2][1], b[3][0], b[3][1]);
            }
            System.out.printf("  [%2d] text=\"%s\"  score=%.6f%s%n", i + 1, r.text(), r.score(), boxStr);
        }

        if (parsed.saveVis != null) {
            saveVis(img, results, Path.of(parsed.saveVis));
        }
    }

    /**
     * 在指定路径绘制检测框与文字并保存为 PNG。
     */
    private static void saveVis(Mat img, List<OCRResult> results, Path out) {
        Mat canvas = img.clone();
        for (OCRResult r : results) {
            Point[] pts = new Point[4];
            for (int i = 0; i < 4; i++) {
                pts[i] = new Point(r.box()[i][0], r.box()[i][1]);
            }
            MatOfPoint mop = new MatOfPoint(pts);
            List<MatOfPoint> list = new ArrayList<>();
            list.add(mop);
            Imgproc.polylines(canvas, list, true, new Scalar(0, 255, 0), 2);
        }
        boolean ok = Imgcodecs.imwrite(out.toAbsolutePath().toString(), canvas);
        if (ok) {
            System.out.println("\n可视化已保存: " + out);
        } else {
            System.err.println("Warning: 保存可视化失败: " + out);
        }
        canvas.release();
    }

    /**
     * 加载 OpenCV 原生库。openpnp 4.10.0-0 自带多平台原生 jar，
     * 第一次调用时自动解压并 loadLibrary。
     */
    private static void loadOpenCV() {
        try {
            nu.pattern.OpenCV.loadShared();
        } catch (Throwable t) {
            System.err.println("无法加载 OpenCV 原生库: " + t.getMessage());
            t.printStackTrace();
            System.exit(1);
        }
    }

    private static void setVerboseLogging() {
        // logback.xml 读取 mica.root.level / mica.ppocr.level 来切换到 DEBUG
        System.setProperty("mica.root.level", "DEBUG");
        System.setProperty("mica.ppocr.level", "DEBUG");
        // 简单日志后端兼容（如未使用 logback）
        System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "debug");
    }

    private static Path findDemoImage() {
        Path cwd = Paths.get("").toAbsolutePath();
        for (String sub : new String[]{"assets", "test_images"}) {
            Path dir = cwd.resolve(sub);
            if (!Files.isDirectory(dir)) continue;
            try (Stream<Path> s = Files.list(dir)) {
                List<Path> pngs = new ArrayList<>();
                s.filter(p -> p.getFileName().toString().toLowerCase().endsWith(".png"))
                        .forEach(pngs::add);
                if (!pngs.isEmpty()) {
                    pngs.sort(Comparator.comparing(p -> p.getFileName().toString()));
                    return pngs.get(0);
                }
            } catch (IOException e) {
                LOG.debug("扫描 {} 失败: {}", dir, e.getMessage());
            }
        }
        return null;
    }

    /** 命令行参数。 */
    static final class Args {
        String image = "test_images/general_ocr_002.png";
        String detModel = "models/PP-OCRv6_tiny_det_onnx/inference.onnx";
        String recModel = "models/PP-OCRv6_tiny_rec_0515_onnx/inference.onnx";
        String dict = "models/rec_char_dict.txt";
        String saveVis = "test_images/output_vis.png";
        boolean verbose;

        static Args parse(String[] argv) {
            Args a = new Args();
            for (int i = 0; i < argv.length; i++) {
                String arg = argv[i];
                switch (arg) {
                    case "--det-model" -> a.detModel = requireValue(argv, ++i, arg);
                    case "--rec-model" -> a.recModel = requireValue(argv, ++i, arg);
                    case "--dict" -> a.dict = requireValue(argv, ++i, arg);
                    case "--save-vis" -> a.saveVis = requireValue(argv, ++i, arg);
                    case "--no-vis" -> a.saveVis = null;
                    case "-v", "--verbose" -> a.verbose = true;
                    case "-h", "--help" -> {
                        printHelp();
                        System.exit(0);
                    }
                    default -> {
                        if (arg.startsWith("-")) {
                            System.err.println("Unknown option: " + arg);
                            printHelp();
                            System.exit(2);
                        }
                        if (a.image != null) {
                            System.err.println("多余的位置参数: " + arg);
                            System.exit(2);
                        }
                        a.image = arg;
                    }
                }
            }
            return a;
        }

        private static String requireValue(String[] argv, int i, String name) {
            if (i >= argv.length) {
                System.err.println("Missing value for " + name);
                System.exit(2);
            }
            return argv[i];
        }

        static void printHelp() {
            System.out.println("""
                    mica-ppocr — PP-OCRv6 纯 ONNX Runtime 推理（Java 17）

                    用法:
                      java -jar mica-ppocr-0.1.0-all.jar [image] [选项]

                    位置参数:
                      image                       输入图片路径（缺省时自动从 assets/ 或 test_images/ 选一个）

                    选项:
                      --det-model PATH            检测 ONNX 模型路径
                      --rec-model PATH            识别 ONNX 模型路径
                      --dict PATH                 字符字典路径
                      --save-vis PATH             保存可视化结果路径（默认 output_vis.png；--no-vis 禁用）
                      -v, --verbose               开启 DEBUG 日志
                      -h, --help                  打印本帮助
                    """);
        }
    }
}
