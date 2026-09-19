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
import net.dreamlu.mica.ai.face.config.MultiFaceStrategy;
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
 *
 * <p>单张图检出多张人脸时按 {@link MultiFaceStrategy} 选脸，默认取面积最大者。
 */
@Getter
public class FaceVerifier {

	private final FaceDetector detector;
	private final FaceAligner aligner;
	private final FeatureExtractor extractor;
	private final float defaultThreshold;
	private final MultiFaceStrategy multiFaceStrategy;

	/**
	 * 使用模型管理器构造比对门面，阈值与多脸策略均取全局配置。
	 *
	 * @param modelManager 模型管理器，提供检测、对齐、特征提取所需资源
	 */
	public FaceVerifier(ModelManager modelManager) {
		this(modelManager, modelManager.getConfig().getMultiFaceStrategy());
	}

	/**
	 * 使用模型管理器构造比对门面，显式指定多脸选脸策略。
	 *
	 * <p>阈值仍取全局配置；策略为 {@code null} 时回退到
	 * {@link MultiFaceStrategy#DEFAULT}。
	 *
	 * @param modelManager      模型管理器，提供检测、对齐、特征提取所需资源
	 * @param multiFaceStrategy 多张人脸时的选脸策略，{@code null} 表示用默认策略
	 */
	public FaceVerifier(ModelManager modelManager, MultiFaceStrategy multiFaceStrategy) {
		this.detector = new FaceDetector(modelManager);
		this.aligner = new FaceAligner();
		this.extractor = new FeatureExtractor(modelManager);
		this.defaultThreshold = modelManager.getConfig().getVerifyThreshold();
		this.multiFaceStrategy = multiFaceStrategy != null
			? multiFaceStrategy : MultiFaceStrategy.DEFAULT;
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
		FaceBox box = selectFace(detector.detect(image), multiFaceStrategy);
		if (box == null) {
			throw new MicaAiException(
				ErrorCode.VERIFICATION_FAILED, "未检测到人脸");
		}
		Mat aligned = null;
		try {
			aligned = aligner.align(image, box);
			return extractor.extract(aligned);
		} finally {
			ImageUtils.releaseAll(aligned);
		}
	}

	/**
	 * 按策略从单张图的检测结果中选出用于比对的那张人脸。
	 *
	 * <p>包级可见以便单测直接覆盖选脸逻辑（无需加载模型）。
	 *
	 * @param boxes    单张图的人脸检测结果，可为空
	 * @param strategy 选脸策略，不可为 {@code null}（由构造器归一化保证）
	 * @return 选定人脸；无任何人脸时返回 {@code null}
	 * @throws MicaAiException {@link ErrorCode#VERIFICATION_FAILED}
	 *                        策略为 {@link MultiFaceStrategy#REJECT} 且检出多于一张人脸
	 */
	static FaceBox selectFace(List<FaceBox> boxes, MultiFaceStrategy strategy) {
		if (boxes == null || boxes.isEmpty()) {
			return null;
		}
		if (boxes.size() == 1) {
			return boxes.get(0);
		}
		switch (strategy) {
			case REJECT:
				throw new MicaAiException(ErrorCode.VERIFICATION_FAILED,
					"检测到 " + boxes.size() + " 张人脸，策略 REJECT 拒绝比对");
			case LARGEST_SCORE:
				return byLargestScore(boxes);
			case LARGEST_AREA:
			default:
				return byLargestArea(boxes);
		}
	}

	private static FaceBox byLargestScore(List<FaceBox> boxes) {
		FaceBox best = boxes.get(0);
		for (int i = 1; i < boxes.size(); i++) {
			if (boxes.get(i).getScore() > best.getScore()) {
				best = boxes.get(i);
			}
		}
		return best;
	}

	private static FaceBox byLargestArea(List<FaceBox> boxes) {
		FaceBox best = boxes.get(0);
		float bestArea = area(best);
		for (int i = 1; i < boxes.size(); i++) {
			float a = area(boxes.get(i));
			if (a > bestArea) {
				bestArea = a;
				best = boxes.get(i);
			}
		}
		return best;
	}

	private static float area(FaceBox box) {
		return (box.getX2() - box.getX1()) * (box.getY2() - box.getY1());
	}
}