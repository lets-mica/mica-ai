/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.plate.pipeline;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.plate.config.PlateConfig;
import net.dreamlu.mica.ai.plate.alignment.PlateAligner;
import net.dreamlu.mica.ai.plate.detection.PlateDetector;
import net.dreamlu.mica.ai.plate.model.PlateResult;
import net.dreamlu.mica.ai.plate.model.PlateType;
import net.dreamlu.mica.ai.plate.recognition.PlateClassifier;
import net.dreamlu.mica.ai.plate.recognition.PlateRecognizer;
import net.dreamlu.mica.ai.plate.util.PlateImageUtils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * 车牌识别门面（HyperLPR3 LPRMultiTaskPipeline 的 Java 复刻）。
 *
 * <p>流程：{@link PlateDetector} 拿到 bbox + 4 关键点 →
 * {@link PlateAligner} 透视校正 →
 * {@link PlateRecognizer} CRNN 识别（双层车牌切成上下两行各跑一次） →
 * {@link PlateClassifier} 仅在首字符无法判定颜色时调用 → 输出 {@link PlateResult}。
 */
@Slf4j
public class PlatePipeline implements AutoCloseable {

	private final PlateDetector detector;
	private final PlateRecognizer recognizer;
	private final PlateClassifier classifier;
	@Getter
	private final int maxPlates;

	public PlatePipeline(PlateConfig config) {
		Objects.requireNonNull(config, "PlateConfig must not be null");
		config.validate();
		if (config.getDetectionModelPath() == null || config.getDetectionModelPath().isEmpty()
			|| config.getRecognitionModelPath() == null || config.getRecognitionModelPath().isEmpty()
			|| config.getClassificationModelPath() == null || config.getClassificationModelPath().isEmpty()) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED,
				"detection / recognition / classification 模型路径必须全部配置");
		}
		OrtEnvironment environment = OrtEnvironment.getEnvironment();
		OrtSession.SessionOptions sessionOptions = buildSessionOptions(config.getOnnx());
		this.detector = new PlateDetector(environment, config, sessionOptions);
		this.recognizer = new PlateRecognizer(environment, config, sessionOptions);
		this.classifier = new PlateClassifier(environment, config, sessionOptions);
		this.maxPlates = config.getMaxPlates();
		log.info("mica-ai-plate 初始化完成: version={} maxPlates={}",
			config.getModelVersion(), maxPlates);
	}

	public static PlatePipeline create(PlateConfig config) {
		return new PlatePipeline(config);
	}

	public static PlatePipeline createDefault() {
		return new PlatePipeline(PlateConfig.builder().build());
	}

	private static OrtSession.SessionOptions buildSessionOptions(OrtSessionOptions onnx) {
		OrtSession.SessionOptions so = new OrtSession.SessionOptions();
		try {
			if (onnx != null) {
				if (onnx.getIntraOpNumThreads() > 0) {
					so.setIntraOpNumThreads(onnx.getIntraOpNumThreads());
				}
				if (onnx.getInterOpNumThreads() > 0) {
					so.setInterOpNumThreads(onnx.getInterOpNumThreads());
				}
				OrtDevice device = onnx.getDevice();
				if (OrtDevice.GPU == device) {
					Set<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
					if (providers != null && providers.contains(OrtProvider.CUDA)) {
						so.addCUDA(onnx.getCudaDeviceId());
						log.info("mica-ai-plate: 已启用 CUDA 执行提供器 (deviceId={})", onnx.getCudaDeviceId());
					} else {
						log.warn("mica-ai-plate: 未检测到可用的 CUDA 执行提供器，回退到 CPU 推理");
					}
				}
			}
		} catch (OrtException e) {
			throw new MicaAiException(ErrorCode.MODEL_LOAD_FAILED, "配置 ONNX 会话选项失败", e);
		}
		return so;
	}

	static PlateType codeFilter(String plateCode) {
		if (plateCode == null || plateCode.isEmpty()) {
			return PlateType.UNKNOWN;
		}
		if (plateCode.startsWith("WJ")) {
			return PlateType.WHITE_POLICE;
		}
		if (plateCode.length() == 8) {
			return PlateType.GREEN;
		}
		if (plateCode.indexOf('学') >= 0) {
			return PlateType.BLUE;
		}
		if (plateCode.indexOf('港') >= 0 || plateCode.indexOf('澳') >= 0) {
			return PlateType.HONG_KONG_MACAO;
		}
		if (plateCode.indexOf('警') >= 0) {
			return PlateType.WHITE_POLICE;
		}
		if (plateCode.startsWith("粤Z")) {
			return PlateType.HONG_KONG_MACAO;
		}
		return PlateType.UNKNOWN;
	}

	public List<PlateResult> recognizePath(String imagePath) {
		Path p = Paths.get(imagePath);
		if (!Files.exists(p)) {
			throw new MicaAiException(ErrorCode.NOT_FOUND, "文件不存在: " + imagePath);
		}
		Mat bgr = null;
		try {
			bgr = PlateImageUtils.byteArrayToMat(Files.readAllBytes(p));
			return recognize(bgr);
		} catch (IOException e) {
			throw new MicaAiException(ErrorCode.UNKNOWN, "读取图像失败: " + imagePath, e);
		} finally {
			PlateImageUtils.releaseAll(bgr);
		}
	}

	public List<PlateResult> recognizeBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = PlateImageUtils.byteArrayToMat(imageBytes);
			return recognize(bgr);
		} finally {
			PlateImageUtils.releaseAll(bgr);
		}
	}

	public List<PlateResult> recognize(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return Collections.emptyList();
		}
		List<PlateDetector.Detection> detections = detector.detect(bgr);
		if (detections.isEmpty()) {
			return Collections.emptyList();
		}
		List<PlateResult> results = new ArrayList<>();
		for (int i = 0; i < detections.size() && results.size() < maxPlates; i++) {
			PlateDetector.Detection d = detections.get(i);
			Mat aligned = PlateAligner.rotateCrop(bgr, d.landmarks);
			try {
				if (d.layerNum == 1) {
					int h = aligned.rows();
					int line = (int) Math.round(h * 0.4);
					Mat top = new Mat(aligned, new org.opencv.core.Rect(0, 0, aligned.cols(), line));
					Mat bottom = new Mat(aligned, new org.opencv.core.Rect(0, line, aligned.cols(), h - line));
					try {
						PlateRecognizer.RecognitionResult topR = recognizer.recognize(top);
						PlateRecognizer.RecognitionResult botR = recognizer.recognize(bottom);
						if (topR.text.isEmpty() && botR.text.isEmpty()) {
							continue;
						}
						String plateCode = topR.text + botR.text;
						float recConf = (topR.confidence + botR.confidence) / 2f;
						PlateResult result = buildResult(plateCode, d, recConf, aligned);
						results.add(result);
					} finally {
						top.release();
						bottom.release();
					}
				} else {
					PlateRecognizer.RecognitionResult r = recognizer.recognize(aligned);
					if (r.text.isEmpty()) {
						continue;
					}
					PlateResult result = buildResult(r.text, d, r.confidence, aligned);
					results.add(result);
				}
			} finally {
				aligned.release();
			}
		}
		return results;
	}

	private PlateResult buildResult(String plateCode, PlateDetector.Detection d,
									float recConfidence, Mat alignedPad) {
		PlateType type = codeFilter(plateCode);
		if (type == PlateType.UNKNOWN && alignedPad != null && !alignedPad.empty()) {
			int cls = classifier.classify(alignedPad);
			switch (cls) {
				case PlateClassifier.YELLOW:
					type = d.layerNum == 1 ? PlateType.YELLOW_DOUBLE : PlateType.YELLOW_SINGLE;
					break;
				case PlateClassifier.BLUE:
					type = PlateType.BLUE;
					break;
				case PlateClassifier.GREEN:
					type = PlateType.GREEN;
					break;
				default:
					type = PlateType.UNKNOWN;
					break;
			}
		}
		return new PlateResult(plateCode, type, d.detectionScore, recConfidence,
			d.boundingBox, d.landmarks);
	}

	@Override
	public void close() {
		if (detector != null) {
			detector.close();
		}
		if (recognizer != null) {
			recognizer.close();
		}
		if (classifier != null) {
			classifier.close();
		}
	}
}
