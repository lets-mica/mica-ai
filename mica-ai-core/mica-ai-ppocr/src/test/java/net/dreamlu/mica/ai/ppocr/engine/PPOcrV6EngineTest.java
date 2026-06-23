package net.dreamlu.mica.ai.ppocr.engine;

import net.dreamlu.mica.ai.ppocr.config.PPOcrV6Config;
import net.dreamlu.mica.ai.ppocr.config.PPOcrV6Result;
import net.dreamlu.mica.ai.ppocr.engine.PPOcrV6Engine;
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
 * PPOcrV6Engine 命令行入口。
 *
 * <p>使用前请先运行 model-tools/ppocr/download.py 下载模型：
 * <pre>
 *   cd model-tools/ppocr
 *   python download.py            # 默认下载 medium 规格
 *   python download.py --spec tiny # 或指定 tiny/small/medium
 *   python convert.py             # 整理目录结构
 * </pre>
 */
public final class PPOcrV6EngineTest {

    private static final Logger LOG = LoggerFactory.getLogger(PPOcrV6EngineTest.class);

    private PPOcrV6EngineTest() {
    }

    public static void main(String[] args) {
        loadOpenCV();

        Args parsed = Args.parse(args);
        if (parsed.verbose) {
            setVerboseLogging();
        }

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
            System.err.println("Error: 模型文件缺失，请先运行 model-tools/ppocr/download.py 下载模型：");
            System.err.println("  cd model-tools/ppocr");
            System.err.println("  python download.py [--spec tiny|small|medium]");
            System.err.println("  python convert.py");
            System.err.println("\n缺失的文件：");
            missing.forEach(System.err::println);
            System.exit(1);
        }

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
        List<PPOcrV6Result> results;
        var config = PPOcrV6Config.builder()
            .detModelPath(parsed.detModel)
            .recModelPath(parsed.recModel)
            .recCharDictPath(parsed.dict)
            .build();
        try (PPOcrV6Engine ocr = new PPOcrV6Engine(config)) {
            System.out.println("Running OCR...");
            results = ocr.run(img);
        }
        long elapsed = System.currentTimeMillis() - t0;
        System.out.println("\n检测到 " + results.size() + " 个文本区域（耗时 " + elapsed + " ms）：\n");
        for (int i = 0; i < results.size(); i++) {
            PPOcrV6Result r = results.get(i);
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

    private static void saveVis(Mat img, List<PPOcrV6Result> results, Path out) {
        Mat canvas = img.clone();
        for (PPOcrV6Result r : results) {
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
        System.setProperty("mica.root.level", "DEBUG");
        System.setProperty("mica.ppocr.level", "DEBUG");
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

    static final class Args {
        String image = "model-tools/test_images/general_ocr_002.png";
        String detModel = "model-tools/ppocr/model/out/det/inference.onnx";
        String recModel = "model-tools/ppocr/model/out/rec/inference.onnx";
        // 默认使用 tiny 路径（与参考项目 ppocrv6_onnx 一致：
        // model PP-OCRv6_tiny_rec_0515_onnx, vocab=7180）。
        // 历史遗留的 rec_char_dict.txt 就是 tiny 字典（md5 与参考项目一致），
        // 早期默认 medium 模型 + tiny 字典会导致严重乱码，已修复。
        // 若想跑 medium/server，请用对应的 PaddleOCR 多语种字典，
        // 务必确保 vocab 行数 + 1 (blank) 与模型输出维度一致。
        String dict = "model-tools/ppocr/model/out/rec_char_dict.txt";
        String saveVis = "model-tools/test_images/output_vis.png";
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
