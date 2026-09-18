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
 *
 * <p>线程安全：内部 detector / recognizer / classifier 均为 stateless + ONNX session
 * 线程安全；本类自身仅含构造时冻结的不可变字段，可作为 Spring 单例 Bean 共享。
 */
@Slf4j
public class PlatePipeline implements AutoCloseable {

	private final PlateDetector detector;
	private final PlateRecognizer recognizer;
	private final PlateClassifier classifier;
	@Getter
	private final int maxPlates;

	/**
	 * 构造车牌识别 Pipeline，加载检测、识别、分类三个模型。
	 *
	 * @param config 车牌识别配置
	 * @throws MicaAiException 模型路径未配置或加载失败时抛出，{@link ErrorCode#MODEL_LOAD_FAILED}
	 */
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

	/**
	 * 使用指定配置创建车牌识别 Pipeline。
	 *
	 * @param config 车牌识别配置
	 * @return Pipeline 实例
	 */
	public static PlatePipeline create(PlateConfig config) {
		return new PlatePipeline(config);
	}

	/**
	 * 使用默认配置创建车牌识别 Pipeline。
	 *
	 * @return Pipeline 实例
	 */
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

	/**
	 * 识别本地图像文件中的车牌。
	 *
	 * @param imagePath 图像文件路径
	 * @return 车牌识别结果列表
	 * @throws MicaAiException 文件不存在或读取失败时抛出，{@link ErrorCode#NOT_FOUND} / {@link ErrorCode#UNKNOWN}
	 */
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

	/**
	 * 识别图像字节数组中的车牌。
	 *
	 * @param imageBytes 图像文件字节（jpg/png 等）
	 * @return 车牌识别结果列表
	 */
	public List<PlateResult> recognizeBytes(byte[] imageBytes) {
		Mat bgr = null;
		try {
			bgr = PlateImageUtils.byteArrayToMat(imageBytes);
			return recognize(bgr);
		} finally {
			PlateImageUtils.releaseAll(bgr);
		}
	}

	/**
	 * 识别 BGR 图像中的车牌。
	 *
	 * @param bgr BGR 格式输入图像
	 * @return 车牌识别结果列表，图像为空或未检测到车牌时返回空列表
	 */
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
