/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.engine;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.config.FaceConfig;
import net.dreamlu.mica.ai.face.config.FaceEmbedding;
import net.dreamlu.mica.ai.face.utils.ImageUtils;

import java.awt.image.BufferedImage;
import java.nio.FloatBuffer;

/**
 * SFace 人脸识别引擎（OpenCV Zoo，纯 ONNX Runtime）。
 *
 * <p>对应模型：{@code face_recognition_sface_2021dec.onnx}，固定输入 112x112。
 *
 * <p>输入图像应当已经被对齐到 112x112（参见
 * {@link ImageUtils#align(BufferedImage, net.dreamlu.mica.ai.face.config.FaceBox, int)}）。
 * 预处理：RGB 顺序、{@code (x - 127.5) / 128} 归一化。
 * 输出：512 维 Embedding，本类内部做 L2 归一化后返回。
 *
 * <p>License：Apache-2.0，可商用。
 *
 * @since 1.0.0
 */
@Slf4j
public class SFaceRecognizer implements FaceRecognizer {

	private final OrtSession session;
	private final FaceConfig config;

	public SFaceRecognizer(FaceConfig config) {
		this.config = config;
		try {
			OrtEnvironment env = OrtEnvironment.getEnvironment();
			OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
			opts.setIntraOpNumThreads(config.getIntraOpNumThreads());
			opts.setInterOpNumThreads(config.getInterOpNumThreads());
			this.session = env.createSession(config.getRecModelPath().toString(), opts);
			log.info("[mica-ai-face] SFace loaded: {}", config.getRecModelPath());
		} catch (OrtException e) {
			throw new MicaAiException("Failed to load SFace model: " + config.getRecModelPath(), e);
		}
	}

	@Override
	public FaceEmbedding extract(BufferedImage alignedFace) {
		try {
			int size = config.getRecInputSize();
			// SFace 预处理：RGB float32，(x - 127.5) / 128
			float[] data = ImageUtils.toNchwFloat(alignedFace, true);
			OnnxTensor tensor = OnnxTensor.createTensor(
				OrtEnvironment.getEnvironment(),
				FloatBuffer.wrap(data),
				new long[]{1, 3, size, size}
			);
			try (OrtSession.Result result = session.run(java.util.Collections.singletonMap("input", tensor))) {
				OnnxTensor out = (OnnxTensor) result.get(0);
				float[][] vec = (float[][]) out.getValue();
				// L2 归一化（结果 = 单位向量，点积 = 余弦相似度）
				return new FaceEmbedding(FaceRecognizer.l2Normalize(vec[0]));
			} finally {
				tensor.close();
			}
		} catch (OrtException e) {
			throw new MicaAiException("SFace inference failed", e);
		}
	}

	@Override
	public void close() {
		try {
			session.close();
		} catch (OrtException e) {
			log.warn("[mica-ai-face] failed to close SFace session", e);
		}
	}
}
