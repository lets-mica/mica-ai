/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face;

import net.dreamlu.mica.ai.face.config.FaceBox;
import net.dreamlu.mica.ai.face.config.FaceConfig;
import net.dreamlu.mica.ai.face.config.FaceEmbedding;
import net.dreamlu.mica.ai.face.engine.*;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Random;

/**
 * mica-ai-face 端到端冒烟测试，可用 {@code java} 命令直接运行：
 * <pre>{@code
 * mvn -pl mica-ai-core/mica-ai-face dependency:build-classpath \
 *     -DincludeScope=test -Dmdep.outputFile=cp.txt -q
 * CP=$(cat cp.txt)
 * java -cp "target/classes:target/test-classes:$CP" \
 *      net.dreamlu.mica.ai.face.FaceEngineSmokeTest
 * }</pre>
 *
 * <p>分两部分：
 * <ol>
 *   <li><b>纯算法 demo</b>（不需要 ONNX 模型）：合成 512d Embedding，演示
 *       {@link FaceRecognizer#l2Normalize(float[])} 与
 *       {@link FaceRecognizer#cosineSimilarity(float[], float[])}</li>
 *   <li><b>完整链路 demo</b>（需要 OpenCV Zoo YuNet + SFace 两个 ONNX）：生成合成测试图，调用
 *       {@link FaceEngine#extract(BufferedImage)} 输出真实 Embedding</li>
 * </ol>
 *
 * <p>模型缺失时给出修复指引，不抛异常退出。
 */
public class FaceEngineSmokeTest {

	/**
	 * OpenCV Zoo 默认模型文件名
	 */
	private static final String DET_MODEL = "face_detection_yunet_2023mar.onnx";
	private static final String REC_MODEL = "face_recognition_sface_2021dec.onnx";
	private static final Path DEFAULT_MODELS_DIR = Path.of("model-tools", "face", "model", "out");

	public static void main(String[] args) throws Exception {
		banner("mica-ai-face 冒烟测试 (OpenCV Zoo YuNet + SFace)");
		System.out.println("JDK: " + System.getProperty("java.version"));
		System.out.println("工作目录: " + Path.of("").toAbsolutePath());

		// 1. 纯算法 demo，不依赖任何 ONNX 模型
		algorithmDemo();

		// 2. 完整链路 demo，需要 ONNX 模型
		fullPipelineDemo(args);
	}

	// ------------------------------------------------------------------
	// 1) 纯算法 demo
	// ------------------------------------------------------------------
	private static void algorithmDemo() {
		banner("Part 1 / 纯算法 demo（不需要 ONNX 模型）");
		Random rnd = new Random(42);

		// 模拟 alice 和 bob 的 512d Embedding
		float[] alice = randomUnitVector(512, rnd);
		float[] bob = randomUnitVector(512, rnd);
		// alice 同一人的另一张照（与 alice 高相似，与 bob 低相似）
		float[] alice2 = perturb(alice, 0.01f, rnd);

		// 演示 L2 normalize：让随机向量变成长度为 1 的单位向量
		float[] raw = {3f, 4f};
		float[] norm = FaceRecognizer.l2Normalize(raw);
		System.out.printf("  L2 normalize: [3, 4] -> [%.3f, %.3f]  (length=%.3f)%n",
			norm[0], norm[1], Math.sqrt(norm[0] * norm[0] + norm[1] * norm[1]));

		// 演示 cosine similarity（SFace 推荐阈值 ~0.363 / 1:1 验证）
		float simAA = FaceRecognizer.cosineSimilarity(alice, alice2);
		float simAB = FaceRecognizer.cosineSimilarity(alice, bob);
		float threshold = 0.5f;
		System.out.printf("  alice  vs alice2 -> %.4f  %s%n", simAA, judge(simAA, threshold));
		System.out.printf("  alice  vs bob    -> %.4f  %s%n", simAB, judge(simAB, threshold));
	}

	// ------------------------------------------------------------------
	// 2) 完整链路 demo
	// ------------------------------------------------------------------
	private static void fullPipelineDemo(String[] args) throws Exception {
		banner("Part 2 / 完整链路 demo（需要 OpenCV Zoo ONNX 模型）");

		Path detPath = resolveArgOrDefault(args, 0, DEFAULT_MODELS_DIR.resolve(DET_MODEL));
		Path recPath = resolveArgOrDefault(args, 1, DEFAULT_MODELS_DIR.resolve(REC_MODEL));

		if (!Files.exists(detPath) || !Files.exists(recPath)) {
			System.out.println("  ⚠  ONNX 模型未找到，跳过真实推理。");
			System.out.println("    期望位置:");
			System.out.println("      " + detPath.toAbsolutePath());
			System.out.println("      " + recPath.toAbsolutePath());
			System.out.println();
			System.out.println("  修复方式：");
			System.out.println("    cd model-tools/face");
			System.out.println("    python download.py        # 下载 OpenCV Zoo YuNet + SFace (~90MB)");
			System.out.println("    python convert.py         # 拷贝到 model/out/");
			System.out.println();
			System.out.println("  或自行下载 OpenCV Zoo 模型后重试，传入模型路径：");
			System.out.println("    java ... FaceEngineSmokeTest /path/to/yunet.onnx /path/to/sface.onnx");
			return;
		}

		// 合成两张测试图（640x640，模拟人脸）
		BufferedImage image1 = syntheticFace(640, 640, "Alice", Color.PINK);
		BufferedImage image2 = syntheticFace(640, 640, "Bob", Color.CYAN);

		// 保存为 PNG（便于人工目检）
		Path tmp = Files.createTempDirectory("mica-ai-face-smoke-");
		Path p1 = tmp.resolve("alice.png");
		Path p2 = tmp.resolve("bob.png");
		ImageIO.write(image1, "png", p1.toFile());
		ImageIO.write(image2, "png", p2.toFile());
		System.out.println("  生成合成测试图:");
		System.out.println("    " + p1);
		System.out.println("    " + p2);

		FaceConfig faceConfig = FaceConfig.builder()
			.detModelPath(detPath)
			.recModelPath(recPath)
			.build();

		System.out.println("  加载 ONNX 模型...");

		try (
			FaceDetector faceDetector = new YuNetDetector(faceConfig);
			FaceRecognizer faceRecognizer = new SFaceRecognizer(faceConfig);
			FaceEngine engine = FaceEngine.builder()
				.detector(faceDetector)
				.recognizer(faceRecognizer)
				.config(faceConfig).build()
		) {
			// 2.1 detect 演示
			List<FaceBox> boxes = engine.detect(image1);
			System.out.printf("%n  detect(image1) -> %d 张人脸%n", boxes.size());
			for (int i = 0; i < boxes.size(); i++) {
				FaceBox b = boxes.get(i);
				System.out.printf("    [face %d] score=%.3f  box=[%.0f,%.0f,%.0f,%.0f]%n",
					i, b.getScore(), b.getX1(), b.getY1(), b.getX2(), b.getY2());
			}

			// 2.2 extract 演示
			List<FaceEmbedding> emb1 = engine.extract(image1);
			List<FaceEmbedding> emb2 = engine.extract(image2);
			if (emb1.isEmpty() || emb2.isEmpty()) {
				System.out.println("  ⚠  合成图未检测到人脸（这是正常的，YuNet 在合成图上常无检出）。");
				System.out.println("    替换为真实人脸照片后重试即可。");
				return;
			}

			FaceEmbedding alice = emb1.get(0);
			FaceEmbedding bob = emb2.get(0);
			System.out.printf("%n  extract(image1)[0] -> dim=%d  L2=%.6f%n",
				alice.dimension(), norm(alice.getVector()));
			System.out.printf("  extract(image2)[0] -> dim=%d  L2=%.6f%n",
				bob.dimension(), norm(bob.getVector()));

			// 2.3 1:1 比对（FaceEngine 仅做检测 + 推理，比对走 FaceRecognizer 静态方法）
			float score = FaceRecognizer.cosineSimilarity(alice.getVector(), bob.getVector());
			System.out.printf("%n  similarity(alice, bob) = %.4f  %s%n", score, judge(score, 0.5f));
		}
	}

	// ------------------------------------------------------------------
	// 工具方法
	// ------------------------------------------------------------------
	private static void banner(String title) {
		System.out.println();
		System.out.println("==[ " + title + " ]==");
	}

	private static Path resolveArgOrDefault(String[] args, int idx, Path def) {
		if (args != null && args.length > idx) {
			return Path.of(args[idx]);
		}
		return def;
	}

	private static float[] randomUnitVector(int dim, Random rnd) {
		float[] v = new float[dim];
		for (int i = 0; i < dim; i++) {
			v[i] = (float) (rnd.nextGaussian());
		}
		return FaceRecognizer.l2Normalize(v);
	}

	/**
	 * 在 unit vector 上叠加高斯噪声，扰动幅度 coef 越大相似度越低。
	 */
	private static float[] perturb(float[] base, float coef, Random rnd) {
		float[] v = base.clone();
		for (int i = 0; i < v.length; i++) {
			v[i] += (float) (coef * rnd.nextGaussian());
		}
		return FaceRecognizer.l2Normalize(v);
	}

	private static float norm(float[] v) {
		double s = 0;
		for (float f : v) {
			s += (double) f * f;
		}
		return (float) Math.sqrt(s);
	}

	private static String judge(float score, float threshold) {
		return score >= threshold ? "✓ 同人" : "✗ 不同人";
	}

	/**
	 * 用 Java2D 合成一张带人脸轮廓的彩色图（仅用于"程序能跑通"演示，不是真实人脸）。
	 */
	private static BufferedImage syntheticFace(int w, int h, String name, Color color) {
		BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_3BYTE_BGR);
		Graphics2D g = img.createGraphics();
		try {
			g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
			g.setColor(new Color(245, 245, 250));
			g.fillRect(0, 0, w, h);
			// 头
			int cx = w / 2, cy = h / 2;
			g.setColor(color);
			g.fillOval(cx - 120, cy - 160, 240, 280);
			// 眼
			g.setColor(Color.WHITE);
			g.fillOval(cx - 70, cy - 60, 30, 22);
			g.fillOval(cx + 40, cy - 60, 30, 22);
			g.setColor(Color.BLACK);
			g.fillOval(cx - 60, cy - 54, 14, 14);
			g.fillOval(cx + 50, cy - 54, 14, 14);
			// 嘴
			g.setColor(new Color(200, 50, 50));
			g.setStroke(new BasicStroke(4f));
			g.drawArc(cx - 50, cy + 10, 100, 60, 0, 180);
			// 标签
			g.setColor(new Color(60, 60, 60));
			g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 28));
			g.drawString(name, 30, 50);
		} finally {
			g.dispose();
		}
		return img;
	}
}
