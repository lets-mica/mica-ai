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
 *
 * <p>线程安全：内部持有的 {@link FaceDetector} / {@link FeatureExtractor} stateless 且
 * ONNX session 线程安全；本类自身仅含构造时冻结的不可变字段，可作为 Spring 单例 Bean。
 */
@Getter
public class FaceVerifier {

	/**
	 * 多张人脸时的选脸策略（保留枚举，当前实现始终按面积最大选；保留以备后续扩展）。
	 */
	public enum MultiFaceStrategy {
		/** 多脸时直接拒绝。 */
		REJECT,
		/** 保留得分最高的人脸。 */
		LARGEST_SCORE,
		/** 保留面积最大的人脸。 */
		LARGEST_AREA
	}

	private final FaceDetector detector;
	private final FaceAligner aligner;
	private final FeatureExtractor extractor;
	private final float defaultThreshold;

	/**
	 * 使用模型管理器构造比对门面，阈值取全局配置。
	 *
	 * @param modelManager 模型管理器，提供检测、对齐、特征提取所需资源
	 */
	public FaceVerifier(ModelManager modelManager) {
		this.detector = new FaceDetector(modelManager);
		this.aligner = new FaceAligner();
		this.extractor = new FeatureExtractor(modelManager);
		this.defaultThreshold = modelManager.getConfig().getVerifyThreshold();
	}

	/**
	 * 计算两张图中人脸特征的余弦相似度。
	 *
	 * @param image1 第一张图像
	 * @param image2 第二张图像
	 * @return 余弦相似度，取值约 [-1, 1]
	 * @throws MicaAiException 任一图像未检测到人脸时抛出，错误码 {@link ErrorCode#VERIFICATION_FAILED}
	 */
	public float similarity(Mat image1, Mat image2) {
		float[] f1 = extractSingle(image1);
		float[] f2 = extractSingle(image2);
		return FeatureExtractor.compare(f1, f2);
	}

	/**
	 * 以指定阈值做 1:1 比对。
	 *
	 * @param probe     待比对图像
	 * @param reference 参考图像
	 * @param threshold 判定阈值，为 {@code null} 时使用默认阈值
	 * @return 比对结果（相似度、阈值与是否通过）
	 * @throws MicaAiException 任一图像未检测到人脸时抛出，错误码 {@link ErrorCode#VERIFICATION_FAILED}
	 */
	public VerifyResult verify(Mat probe, Mat reference, Float threshold) {
		float[] f1 = extractSingle(probe);
		float[] f2 = extractSingle(reference);
		float sim = FeatureExtractor.compare(f1, f2);
		float th = threshold != null ? threshold : defaultThreshold;
		return new VerifyResult(sim, th, sim >= th);
	}

	/**
	 * 使用默认阈值做 1:1 比对。
	 *
	 * @param probe     待比对图像
	 * @param reference 参考图像
	 * @return 比对结果（相似度、阈值与是否通过）
	 * @throws MicaAiException 任一图像未检测到人脸时抛出，错误码 {@link ErrorCode#VERIFICATION_FAILED}
	 */
	public VerifyResult verify(Mat probe, Mat reference) {
		return verify(probe, reference, null);
	}

	/**
	 * 对图像字节数组以指定阈值做 1:1 比对。
	 *
	 * @param probeBytes     待比对图像字节数组（JPEG/PNG 等编码格式）
	 * @param referenceBytes 参考图像字节数组（JPEG/PNG 等编码格式）
	 * @param threshold      判定阈值，为 {@code null} 时使用默认阈值
	 * @return 比对结果（相似度、阈值与是否通过）
	 * @throws MicaAiException 解码失败或任一图像未检测到人脸时抛出，错误码 {@link ErrorCode#VERIFICATION_FAILED}
	 */
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

	/**
	 * 对图像字节数组使用默认阈值做 1:1 比对。
	 *
	 * @param probeBytes     待比对图像字节数组（JPEG/PNG 等编码格式）
	 * @param referenceBytes 参考图像字节数组（JPEG/PNG 等编码格式）
	 * @return 比对结果（相似度、阈值与是否通过）
	 * @throws MicaAiException 解码失败或任一图像未检测到人脸时抛出，错误码 {@link ErrorCode#VERIFICATION_FAILED}
	 */
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