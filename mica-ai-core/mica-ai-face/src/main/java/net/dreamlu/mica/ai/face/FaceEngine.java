/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.face.config.FaceBox;
import net.dreamlu.mica.ai.face.config.FaceConfig;
import net.dreamlu.mica.ai.face.config.FaceEmbedding;
import net.dreamlu.mica.ai.face.engine.ArcFaceRecognizer;
import net.dreamlu.mica.ai.face.engine.RetinaFaceDetector;
import net.dreamlu.mica.ai.face.preprocess.ImageUtils;

import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Face 识别引擎门面：检测 → 关键点对齐 → 512 维 Embedding。
 *
 * <p><b>本引擎只负责"图片 → 向量"的推理，不包含人脸库 / 1:N 检索能力。</b>
 * 拿到 {@link FaceEmbedding} 后，请使用向量数据库（Milvus / pgvector / Qdrant 等）做入库与检索。
 *
 * <p>典型用法：
 * <pre>{@code
 * try (FaceEngine engine = FaceEngine.builder()
 *     .detModelPath(Path.of("models/det_10g.onnx"))
 *     .recModelPath(Path.of("models/w600k_r50.onnx"))
 *     .build()) {
 *
 *     // 1. 提取所有的人脸向量
 *     List<FaceEmbedding> faces = engine.extract("group.jpg");
 *
 *     // 2. 后续入库与检索由调用方决定
 *     for (FaceEmbedding fe : faces) {
 *         milvusClient.insert(collection, fe.getVector(), userId);
 *     }
 * }
 * }</pre>
 *
 * @since 1.0.0
 */
@Slf4j
public class FaceEngine implements Closeable, AutoCloseable {

	private final FaceConfig config;
	private final RetinaFaceDetector detector;
	private final ArcFaceRecognizer recognizer;

	protected FaceEngine(FaceConfig config) {
		this.config = config;
		this.detector = new RetinaFaceDetector(config);
		this.recognizer = new ArcFaceRecognizer(config);
	}

	/**
	 * Builder 入口，方便链式调用。
	 */
	public static Builder builder() {
		return new Builder();
	}

	/**
	 * 用已有 config 构造（Spring Boot Starter 场景）。
	 */
	public static FaceEngine create(FaceConfig config) {
		return new FaceEngine(config);
	}

	/**
	 * 检测 + 识别：输入一张图，返回图中所有人脸的 Embedding（按 score 降序）。
	 */
	public List<FaceEmbedding> extract(BufferedImage image) {
		List<FaceBox> boxes = detector.detect(image);
		List<FaceEmbedding> result = new ArrayList<>(boxes.size());
		for (FaceBox box : boxes) {
			result.add(recognizer.recognize(image, box));
		}
		return result;
	}

	/**
	 * 便捷重载：从文件路径加载。
	 */
	public List<FaceEmbedding> extract(Path imagePath) throws IOException {
		return extract(ImageUtils.read(imagePath));
	}

	/**
	 * 检测：只返回人脸框（不下游推理）。
	 */
	public List<FaceBox> detect(BufferedImage image) {
		return detector.detect(image);
	}

	public FaceConfig getConfig() {
		return config;
	}

	@Override
	public void close() {
		try {
			detector.close();
		} finally {
			recognizer.close();
		}
	}

	// ------------------------------------------------------------------------
	// Builder
	// ------------------------------------------------------------------------

	public static class Builder {
		private final FaceConfig.FaceConfigBuilder configBuilder = FaceConfig.builder();

		public Builder detModelPath(Path path) {
			configBuilder.detModelPath(path);
			return this;
		}

		public Builder recModelPath(Path path) {
			configBuilder.recModelPath(path);
			return this;
		}

		public Builder detScoreThreshold(float v) {
			configBuilder.detScoreThreshold(v);
			return this;
		}

		public Builder detNmsThreshold(float v) {
			configBuilder.detNmsThreshold(v);
			return this;
		}

		public Builder intraOpNumThreads(int v) {
			configBuilder.intraOpNumThreads(v);
			return this;
		}

		public Builder interOpNumThreads(int v) {
			configBuilder.interOpNumThreads(v);
			return this;
		}

		public FaceEngine build() {
			return new FaceEngine(configBuilder.build());
		}
	}
}
