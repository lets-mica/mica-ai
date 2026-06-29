/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.config.FaceBox;
import net.dreamlu.mica.ai.face.config.FaceConfig;
import net.dreamlu.mica.ai.face.config.FaceEmbedding;
import net.dreamlu.mica.ai.face.engine.FaceDetector;
import net.dreamlu.mica.ai.face.engine.FaceRecognizer;
import net.dreamlu.mica.ai.face.preprocess.ImageUtils;

import java.awt.image.BufferedImage;
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
 * <p>默认实现（OpenCV Zoo YuNet + SFace，Apache-2.0，可商用）：
 * <pre>{@code
 * try (FaceEngine engine = FaceEngine.builder()
 *     .detModelPath(Path.of("models/face_detection_yunet_2023mar.onnx"))
 *     .recModelPath(Path.of("models/face_recognition_sface_2021dec.onnx"))
 *     .build()) {
 *
 *     List<FaceEmbedding> faces = engine.extract("group.jpg");
 * }
 * }</pre>
 *
 * <p>要切换检测/识别实现，注入自定义 {@link FaceDetector} / {@link FaceRecognizer} 即可：
 * <pre>{@code
 * FaceEngine engine = FaceEngine.builder()
 *     .detector(myCustomDetector)
 *     .recognizer(myCustomRecognizer)
 *     .build();
 * }</pre>
 *
 * @since 1.0.0
 */
@Slf4j
@Getter
@RequiredArgsConstructor
public class FaceEngine implements AutoCloseable {
	private final FaceConfig config;
	private final FaceDetector detector;
	private final FaceRecognizer recognizer;

	/**
	 * Builder 入口，方便链式调用。
	 */
	public static Builder builder() {
		return new Builder();
	}

	/**
	 * 用 config 创建默认实现的引擎（YuNet + SFace）。
	 */
	public static FaceEngine create(FaceConfig config) {
		return builder().config(config).build();
	}

	/**
	 * 检测 + 识别：输入一张图，返回图中所有人脸的 Embedding（按 score 降序）。
	 */
	public List<FaceEmbedding> extract(BufferedImage image) {
		List<FaceBox> boxes = detector.detect(image);
		List<FaceEmbedding> result = new ArrayList<>(boxes.size());
		int size = config.getRecInputSize();
		for (FaceBox box : boxes) {
			BufferedImage aligned = ImageUtils.align(image, box, size);
			result.add(recognizer.extract(aligned));
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
		private FaceConfig config;
		private FaceDetector detector;
		private FaceRecognizer recognizer;
		private FaceConfig.FaceConfigBuilder configBuilder;

		/**
		 * 直接注入 detector（最常用：自定义实现）。
		 */
		public Builder detector(FaceDetector detector) {
			this.detector = detector;
			return this;
		}

		/**
		 * 直接注入 recognizer。
		 */
		public Builder recognizer(FaceRecognizer recognizer) {
			this.recognizer = recognizer;
			return this;
		}

		/**
		 * 用整个 config 一次性设置。
		 */
		public Builder config(FaceConfig config) {
			this.config = config;
			this.configBuilder = null;
			return this;
		}

		public FaceEngine build() {
			// 如果用了 configBuilder，先 build 出 config
			if (configBuilder != null) {
				this.config = configBuilder.build();
			}
			if (config == null) {
				throw new MicaAiException("FaceConfig is required (set via .config(...) or .detModelPath(...))");
			}
			if (config.getDetModelPath() == null || config.getRecModelPath() == null) {
				throw new MicaAiException("detModelPath and recModelPath are required");
			}
			return new FaceEngine(config, detector, recognizer);
		}

	}
}
