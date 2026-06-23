package net.dreamlu.mica.ai.ppocr.engine;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.onnx.OrtProviders;
import net.dreamlu.mica.ai.ppocr.config.PPOcrV6Config;
import net.dreamlu.mica.ai.ppocr.config.PPOcrV6Result;
import net.dreamlu.mica.ai.ppocr.postprocessor.CtcLabelDecoder;
import net.dreamlu.mica.ai.ppocr.postprocessor.DbPostProcessor;
import net.dreamlu.mica.ai.ppocr.preprocessor.DetectionPreprocessor;
import net.dreamlu.mica.ai.ppocr.preprocessor.RecognitionPreprocessor;
import net.dreamlu.mica.ai.ppocr.utils.BoxUtil;
import net.dreamlu.mica.ai.ppocr.utils.CropUtil;
import net.dreamlu.mica.ai.ppocr.utils.NdArrayUtils;
import org.opencv.core.Mat;

import java.io.Closeable;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * PP-OCRv6 纯 ONNX Runtime 推理引擎。
 *
 * <p>典型用法：
 * <pre>{@code
 * var config = PPOcrV6Config.builder()
 *     .detModelPath("det.onnx")
 *     .recModelPath("rec.onnx")
 *     .recCharDictPath("dict.txt")
 *     .build();
 * try (var engine = new PPOcrV6Engine(config)) {
 *     List<PPOcrV6Result> results = engine.run(image);
 * }
 * }</pre>
 */
@Slf4j
public final class PPOcrV6Engine implements Closeable, AutoCloseable {

	private final OrtEnvironment env;
	private final OrtSession detSession;
	private final OrtSession recSession;
	private final String detInputName;
	private final String recInputName;

	private final DetectionPreprocessor detPre;
	private final DbPostProcessor detPost;
	private final RecognitionPreprocessor recPre;
	private final CtcLabelDecoder recPost;
	private final int recBatchSize;

	private boolean closed = false;

	public PPOcrV6Engine(PPOcrV6Config config) {
		requireFile(config.getDetModelPath(), "detModelPath");
		requireFile(config.getRecModelPath(), "recModelPath");
		requireFile(config.getRecCharDictPath(), "recCharDictPath");
		if (config.getRecBatchSize() < 1) {
			throw new IllegalArgumentException("recBatchSize must be >= 1, got " + config.getRecBatchSize());
		}
		if (config.getRecImageShape() == null || config.getRecImageShape().length != 3) {
			throw new IllegalArgumentException("recImageShape must be [C, H, W]");
		}
		String[] providers = OrtProviders.resolve(!config.isPreferAccelerator());
		log.info("ONNX Runtime provider: {}", String.join(",", providers));
		this.env = OrtEnvironment.getEnvironment();

		OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
		try {
			opts.setIntraOpNumThreads(Math.max(1, config.getIntraOpNumThreads()));
			opts.setInterOpNumThreads(Math.max(1, config.getInterOpNumThreads()));
		} catch (OrtException e) {
			log.warn("设置线程数失败，使用默认值: {}", e.getMessage());
		}

		try {
			this.detSession = env.createSession(config.getDetModelPath(), opts);
			this.recSession = env.createSession(config.getRecModelPath(), opts);
		} catch (OrtException e) {
			throw new RuntimeException("创建 ONNX Runtime 会话失败: " + e.getMessage(), e);
		}

		this.detInputName = detSession.getInputNames().iterator().next();
		this.recInputName = recSession.getInputNames().iterator().next();

		this.detPre = new DetectionPreprocessor(config.getDetLimitSideLen(), config.getDetLimitType(), config.getDetMaxSideLimit());
		this.detPost = new DbPostProcessor(config.getDetThresh(), config.getDetBoxThresh(), config.getDetUnclipRatio(),
				1000, 3);
		this.recPre = new RecognitionPreprocessor(config.getRecImageShape()[1], 320, 3200);
		this.recPost = new CtcLabelDecoder(config.getRecCharDictPath());
		this.recBatchSize = config.getRecBatchSize();

		log.info("PPOcrV6Engine 初始化完成: det={}, rec={}, vocab={}",
			this.detPre, this.recPre, this.recPost.vocabSize());
	}

	private static void requireFile(String path, String name) {
		if (path == null) {
			throw new IllegalArgumentException(name + " is null");
		}
		if (!Files.isRegularFile(Path.of(path))) {
			throw new IllegalArgumentException(name + ": file not found: " + path);
		}
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
			log.info("PPOcrV6Engine 已关闭");
		}
	}

	private void requireOpen() {
		if (closed) {
			throw new IllegalStateException("PPOcrV6Engine has been closed and can no longer be used.");
		}
	}

	@Override
	public String toString() {
		return "PPOcrV6Engine(det=" + detPre + ", rec=" + recPre
			+ ", vocab=" + recPost.vocabSize() + ", closed=" + closed + ")";
	}

	/**
	 * 文本检测。
	 *
	 * @param imgBgr BGR 格式图像 (H, W, 3) uint8
	 * @return boxes 形状 (N, 4, 2) int，scores 长度 N
	 */
	public DetectResult detect(Mat imgBgr) {
		requireOpen();
		DetectionPreprocessor.Result prep = detPre.call(imgBgr);
		long[] detShape = toLongArray(prep.shape());
		FloatBuffer buf = NdArrayUtils.toBuffer(prep.data());
		Map<String, OnnxTensor> inputs = Collections.singletonMap(detInputName, tensor(buf, detShape));
		try (OrtSession.Result result = detSession.run(inputs)) {
			OnnxTensor outTensor = (OnnxTensor) result.get(0);
			float[][] prob = readProb2D(outTensor);
			DbPostProcessor.Result post = detPost.call(probToMat(prob, prep.imgShape()), prep.imgShape());
			return new DetectResult(post.boxes(), post.scores());
		} catch (OrtException e) {
			throw new RuntimeException("det 推理失败: " + e.getMessage(), e);
		}
	}

	/**
	 * 文本识别（支持批量）。
	 *
	 * @param imgList 裁剪后的 BGR 文本行图像列表
	 * @return texts 与 scores 长度一致
	 */
	public RecognizeResult recognize(List<Mat> imgList) {
		requireOpen();
		int n = imgList.size();
		if (n == 0) {
			return new RecognizeResult(new String[0], new float[0]);
		}
		if (log.isDebugEnabled()) {
			Mat first = imgList.get(0);
			log.debug("rec 输入 #0: {}x{}x{} type={} (BGR)", first.rows(), first.cols(), first.channels(), first.type());
		}

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
			RecognitionPreprocessor.Result prep = recPre.call(batch);
			long[] shape = toLongArray(prep.shape());
			FloatBuffer buf = NdArrayUtils.toBuffer(prep.data());
			Map<String, OnnxTensor> inputs = Collections.singletonMap(recInputName, tensor(buf, shape));
			try (OrtSession.Result result = recSession.run(inputs)) {
				OnnxTensor outTensor = (OnnxTensor) result.get(0);
				float[][][] modelOutput = read3D(outTensor);
				CtcLabelDecoder.Result decoded = recPost.call(modelOutput);
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

	/**
	 * 完整 OCR 流程：检测 → 排序 → 裁剪 → 识别。
	 *
	 * @param imgBgr BGR 格式图像 (H, W, 3) uint8
	 * @return 识别结果列表（按阅读顺序排列）
	 */
	public List<PPOcrV6Result> run(Mat imgBgr) {
		requireOpen();
		DetectResult dr = detect(imgBgr);
		if (dr.boxes().length == 0) {
			return List.of();
		}

		int[][][] sortedBoxes = BoxUtil.sortQuadBoxes(dr.boxes());

		List<Mat> crops = CropUtil.cropByPolys(imgBgr, sortedBoxes);

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

		RecognizeResult rr = recognize(validCrops);

		List<PPOcrV6Result> results = new ArrayList<>(validBoxes.size());
		for (int i = 0; i < validBoxes.size(); i++) {
			results.add(new PPOcrV6Result(rr.texts()[i], rr.scores()[i], validBoxes.get(i)));
		}
		return results;
	}

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

	private float[][] readProb2D(OnnxTensor tensor) throws OrtException {
		FloatBuffer buf = tensor.getFloatBuffer();
		long[] shape = tensor.getInfo().getShape();
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
	// 内部记录
	// ==================================================================

	public record DetectResult(int[][][] boxes, float[] scores) {}

	public record RecognizeResult(String[] texts, float[] scores) {}
}
