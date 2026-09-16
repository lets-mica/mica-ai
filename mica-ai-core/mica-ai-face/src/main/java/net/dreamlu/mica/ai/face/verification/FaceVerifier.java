/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.verification;

import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.alignment.FaceAligner;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.face.recognition.FeatureExtractor;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;

import java.util.List;

/**
 * 人脸 1:1 比对（人证核验）门面。
 *
 * <p>编排完整链路：对两张图分别执行 检测 → 对齐 → 特征提取 → 余弦相似度。
 */
@Getter
public class FaceVerifier {

	public enum MultiFaceStrategy {
		REJECT,
		LARGEST_SCORE,
		LARGEST_AREA
	}

	private final FaceDetector detector;
	private final FaceAligner aligner;
	private final FeatureExtractor extractor;
	private final float defaultThreshold;

	public FaceVerifier(ModelManager modelManager) {
		this.detector = new FaceDetector(modelManager);
		this.aligner = new FaceAligner();
		this.extractor = new FeatureExtractor(modelManager);
		this.defaultThreshold = modelManager.getConfig().getVerifyThreshold();
	}

	public float similarity(Mat image1, Mat image2) {
		float[] f1 = extractSingle(image1);
		float[] f2 = extractSingle(image2);
		return FeatureExtractor.compare(f1, f2);
	}

	public VerifyResult verify(Mat probe, Mat reference, Float threshold) {
		float[] f1 = extractSingle(probe);
		float[] f2 = extractSingle(reference);
		float sim = FeatureExtractor.compare(f1, f2);
		float th = threshold != null ? threshold : defaultThreshold;
		return new VerifyResult(sim, th, sim >= th);
	}

	public VerifyResult verify(Mat probe, Mat reference) {
		return verify(probe, reference, null);
	}

	public VerifyResult verify(byte[] probeBytes, byte[] referenceBytes, Float threshold) {
		Mat p = null;
		Mat r = null;
		try {
			p = ImageUtils.byteArrayToMat(probeBytes);
			r = ImageUtils.byteArrayToMat(referenceBytes);
			return verify(p, r, threshold);
		} finally {
			ImageUtils.releaseAll(p, r);
		}
	}

	public VerifyResult verify(byte[] probeBytes, byte[] referenceBytes) {
		return verify(probeBytes, referenceBytes, null);
	}

	private float[] extractSingle(Mat image) {
		List<FaceBox> boxes = detector.detect(image);
		if (boxes.isEmpty()) {
			throw new MicaAiException(
				ErrorCode.VERIFICATION_FAILED, "未检测到人脸");
		}
		FaceBox box = boxes.get(0);
		if (boxes.size() > 1) {
			float bestArea = area(box);
			for (int i = 1; i < boxes.size(); i++) {
				float a = area(boxes.get(i));
				if (a > bestArea) {
					bestArea = a;
					box = boxes.get(i);
				}
			}
		}
		Mat aligned = null;
		try {
			aligned = aligner.align(image, box);
			return extractor.extract(aligned);
		} finally {
			ImageUtils.releaseAll(aligned);
		}
	}

	private static float area(FaceBox box) {
		return (box.getX2() - box.getX1()) * (box.getY2() - box.getY1());
	}
}