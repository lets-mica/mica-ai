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
package net.dreamlu.mica.ai.example.pipeline.idphoto;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.face.alignment.FaceAligner;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.quality.FaceQualityAssessor;
import net.dreamlu.mica.ai.face.quality.FaceQualityResult;
import net.dreamlu.mica.ai.humanseg.core.HumanSegEngine;
import net.dreamlu.mica.ai.matting.core.MattingEngine;
import net.dreamlu.mica.ai.matting.util.MattingImageUtils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * 证件照一键生成流水线。
 *
 * <p>把一张随手拍的人像照片，变成符合规格的证件照：
 * <ol>
 *   <li><b>定位人脸</b>——{@code mica-ai-face} 的 YuNet 检人脸，多脸取面积最大的一张</li>
 *   <li><b>质量评估</b>——{@code mica-ai-face} 的 eDifFIQA 评估人脸质量，
 *       分数始终回填到结果里；是否因质量不达标而拒绝出片，由
 *       {@link IdPhotoOptions#isQualityGate()} 决定（默认不拦截）</li>
 *   <li><b>抠像</b>——{@code mica-ai-matting}（U²-Net）或 {@code mica-ai-humanseg} 出人像掩码，
 *       合成到纯色底</li>
 *   <li><b>按规格裁剪</b>——按证件照通用规范摆放头部：头顶留白 7%、头高占比 75%、
 *       下巴留白 18%，水平居中于头部</li>
 *   <li><b>超分</b>——{@code mica-ai-enhance} 输出 4 倍尺寸，供印刷</li>
 * </ol>
 *
 * <p><b>越界补齐</b>：裁剪框超出原图的部分不会拉伸变形，而是用底色补齐——
 * 这正是证件照的真实处理方式（背景本来就是纯色）。
 *
 * <p>本类位于 example 模块，作为参考实现；业务模块可按需复制改造。
 *
 * @author DreamLu
 */
@Slf4j
public class IdPhotoPipeline {

	/** 人脸检测器 */
	private final FaceDetector detector;

	/** 人脸对齐器 */
	private final FaceAligner aligner;

	/** 人脸质量评估器，可为 null（未启用质量模型时跳过门控） */
	private final FaceQualityAssessor qualityAssessor;

	/** 通用抠图引擎，可为 null */
	private final MattingEngine mattingEngine;

	/** 人像分割引擎，可为 null */
	private final HumanSegEngine humanSegEngine;

	/** 图像超分引擎，可为 null */
	private final EnhanceEngine enhanceEngine;

	public IdPhotoPipeline(FaceDetector detector,
						   FaceAligner aligner,
						   FaceQualityAssessor qualityAssessor,
						   MattingEngine mattingEngine,
						   HumanSegEngine humanSegEngine,
						   EnhanceEngine enhanceEngine) {
		this.detector = detector;
		this.aligner = aligner;
		this.qualityAssessor = qualityAssessor;
		this.mattingEngine = mattingEngine;
		this.humanSegEngine = humanSegEngine;
		this.enhanceEngine = enhanceEngine;
	}

	/**
	 * 生成证件照。
	 *
	 * @param imageBytes 原图字节
	 * @param options    规格 / 底色 / 门控等参数
	 * @return 生成结果；开启质量门控且质量不达标时 {@code passed=false} 且无成片
	 */
	public IdPhotoResult process(byte[] imageBytes, IdPhotoOptions options) {
		IdPhotoSpec spec = options.getSpec() == null ? IdPhotoSpec.ONE : options.getSpec();
		IdPhotoBackground background = options.getBackground() == null
			? IdPhotoBackground.WHITE : options.getBackground();
		IdPhotoMaskSource maskSource = options.getMaskSource() == null
			? IdPhotoMaskSource.MATTING : options.getMaskSource();

		List<Mat> alive = new ArrayList<>();
		Mat source = null;
		try {
			source = MattingImageUtils.byteArrayToMat(imageBytes);
			alive.add(source);

			// ---- 1. 定位人脸（多脸取面积最大） ----
			List<FaceBox> faces = detector.detect(source);
			if (faces == null || faces.isEmpty()) {
				return failure(false, "NO_FACE", "未检测到人脸，请正对镜头重拍", -1f, null, 0f, null,
					spec, background, maskSource, null);
			}
			FaceBox face = largest(faces);

			// ---- 2. 质量评估（评估始终做；拦不拦由 qualityGate 决定） ----
			float qualityScore = -1f;
			Boolean qualityPassed = null;
			if (qualityAssessor != null) {
				Mat aligned = aligner.align(source, face);
				alive.add(aligned);
				FaceQualityResult quality = qualityAssessor.assess(aligned);
				qualityScore = quality.getScore();
				qualityPassed = quality.passed(options.getMinQuality());
				if (options.isQualityGate() && !qualityPassed) {
					return failure(false, "LOW_QUALITY",
						"人脸质量分 " + round(qualityScore) + " 低于门限 "
							+ options.getMinQuality() + "，请重新拍摄（正对镜头、光线均匀）",
						qualityScore, qualityPassed, face.getScore(), box(face),
						spec, background, maskSource, null);
				}
			}

			// ---- 3. 抠像 ----
			Mat alpha = maskSource == IdPhotoMaskSource.HUMANSEG
				? (humanSegEngine == null ? null : humanSegEngine.alpha(source))
				: (mattingEngine == null ? null : mattingEngine.alpha(source));
			if (alpha == null) {
				return failure(false, "MASK_FAILED",
					"掩码来源 " + maskSource.getLabel() + " 不可用（对应能力未启用）",
					qualityScore, qualityPassed, face.getScore(), box(face),
					spec, background, maskSource, null);
			}
			alive.add(alpha);

			// ---- 4. 按规格裁剪 ----
			float[] faceRect = faceBox(face);
			int[] crop = computeCropRect(faceRect, options);
			Mat canvas = composeIdPhoto(source, alpha, faceRect, options);
			alive.add(canvas);

			// ---- 5. 超分 ----
			Mat output = canvas;
			int scale = 1;
			if (options.isUpscale() && enhanceEngine != null) {
				Mat upscaled = enhanceEngine.upscale(canvas);
				alive.add(upscaled);
				output = upscaled;
				scale = canvas.cols() > 0 ? Math.max(1, output.cols() / canvas.cols()) : 1;
			}

			byte[] png = MattingImageUtils.imencodePng(output);
			String hint = faces.size() > 1
				? ("检测到 " + faces.size() + " 张人脸，已取面积最大的一张出片")
				: "证件照生成完成";
			if (Boolean.FALSE.equals(qualityPassed)) {
				hint += "；人脸质量分 " + round(qualityScore) + " 低于门限 "
					+ options.getMinQuality() + "，未开启质量门控，仍已出片";
			}
			return new IdPhotoResult(true, "OK", hint, png, output.cols(), output.rows(),
				qualityScore, qualityPassed, face.getScore(), box(face),
				spec.getLabel(), background.getLabel(), maskSource.getLabel(),
				crop, scale > 1, scale);
		} finally {
			for (Mat mat : alive) {
				if (mat != null) {
					mat.release();
				}
			}
		}
	}

	/**
	 * 把「人像 + 掩码」合成到纯色底，并按证件照规范裁成规格尺寸。
	 *
	 * <p>这是流水线里唯一与模型无关的纯几何步骤，单独暴露出来便于用确定性合成图做单测：
	 * 给定人脸框与掩码，成片尺寸、头部位置、留白比例都是可精确断言的。
	 *
	 * @param bgr      原始人像图（BGR）
	 * @param alpha    与原图同尺寸的人像掩码（CV_32FC1，取值 0~1）
	 * @param faceBox  人脸框 {@code [x1, y1, x2, y2]}（原图坐标）
	 * @param options  规格 / 底色 / 留白比例等参数
	 * @return 规格尺寸的成片（纯色底），调用方负责 release
	 */
	public static Mat composeIdPhoto(Mat bgr, Mat alpha, float[] faceBox, IdPhotoOptions options) {
		IdPhotoSpec spec = options.getSpec() == null ? IdPhotoSpec.ONE : options.getSpec();
		IdPhotoBackground background = options.getBackground() == null
			? IdPhotoBackground.WHITE : options.getBackground();
		int targetWidth = spec.widthPx(options.getDpi());
		int targetHeight = spec.heightPx(options.getDpi());
		int[] rgb = background.rgb();

		Mat canvas = new Mat(targetHeight, targetWidth, CvType.CV_8UC3,
			new Scalar(rgb[2], rgb[1], rgb[0]));
		try {
			int[] crop = computeCropRect(faceBox, options);
			blitCrop(bgr, alpha, canvas, crop, targetWidth, targetHeight, rgb);
			return canvas;
		} catch (RuntimeException e) {
			canvas.release();
			throw e;
		}
	}

	/**
	 * 计算裁剪矩形（源图坐标）。
	 *
	 * <p>几何依据：头高 = 脸高 × {@code headToFaceRatio}，成片高 = 头高 ÷
	 * {@code headHeightRatio}，头顶再上浮 {@code topMarginRatio} 作为留白；
	 * 成片宽按规格比例换算，水平居中于头部（以人脸中心近似头部中心）。
	 *
	 * @param faceBox 人脸框 {@code [x1, y1, x2, y2]}
	 * @param options 参数
	 * @return 裁剪矩形 {@code [x1, y1, x2, y2]}，可能超出原图边界
	 */
	public static int[] computeCropRect(float[] faceBox, IdPhotoOptions options) {
		IdPhotoSpec spec = options.getSpec() == null ? IdPhotoSpec.ONE : options.getSpec();
		float faceHeight = faceBox[3] - faceBox[1];
		float headHeight = faceHeight * options.getHeadToFaceRatio();
		float cropHeight = headHeight / options.getHeadHeightRatio();
		float cropWidth = cropHeight * spec.aspect();

		float headTop = faceBox[1] - faceHeight * 0.5f;
		float cropTop = headTop - cropHeight * options.getTopMarginRatio();
		float centerX = (faceBox[0] + faceBox[2]) / 2f;
		float cropLeft = centerX - cropWidth / 2f;

		return new int[]{
			Math.round(cropLeft),
			Math.round(cropTop),
			Math.round(cropLeft + cropWidth),
			Math.round(cropTop + cropHeight)};
	}

	/**
	 * 把源图与掩码的交集区域合成到纯色底，再缩放贴进画布。
	 *
	 * <p>只处理交集区域，避免对整张大图做浮点合成；画布先铺满底色，
	 * 因此越界部分天然是底色，与「先合成整图再裁剪」结果一致但省内存。
	 */
	private static void blitCrop(Mat source, Mat alpha, Mat canvas,
								 int[] crop, int targetWidth, int targetHeight, int[] rgb) {
		int cropWidth = crop[2] - crop[0];
		int cropHeight = crop[3] - crop[1];
		if (cropWidth <= 0 || cropHeight <= 0) {
			return;
		}
		int ix1 = Math.max(0, crop[0]);
		int iy1 = Math.max(0, crop[1]);
		int ix2 = Math.min(source.cols(), crop[2]);
		int iy2 = Math.min(source.rows(), crop[3]);
		if (ix2 <= ix1 || iy2 <= iy1) {
			return;
		}

		Mat subSource = source.submat(iy1, iy2, ix1, ix2);
		Mat subAlpha = alpha.submat(iy1, iy2, ix1, ix2);
		Mat composed = null;
		Mat resized = null;
		try {
			composed = MattingImageUtils.compositeOnColor(subSource, subAlpha, rgb);

			float scaleX = (float) targetWidth / cropWidth;
			float scaleY = (float) targetHeight / cropHeight;
			int dx1 = Math.round((ix1 - crop[0]) * scaleX);
			int dy1 = Math.round((iy1 - crop[1]) * scaleY);
			int dx2 = Math.round((ix2 - crop[0]) * scaleX);
			int dy2 = Math.round((iy2 - crop[1]) * scaleY);
			int width = Math.max(1, dx2 - dx1);
			int height = Math.max(1, dy2 - dy1);
			if (dx1 < 0) {
				dx1 = 0;
			}
			if (dy1 < 0) {
				dy1 = 0;
			}
			if (dx1 + width > canvas.cols()) {
				width = canvas.cols() - dx1;
			}
			if (dy1 + height > canvas.rows()) {
				height = canvas.rows() - dy1;
			}
			if (width <= 0 || height <= 0) {
				return;
			}

			resized = new Mat();
			Imgproc.resize(composed, resized, new Size(width, height));
			Mat dst = canvas.submat(dy1, dy1 + height, dx1, dx1 + width);
			resized.copyTo(dst);
		} finally {
			if (composed != null) {
				composed.release();
			}
			if (resized != null) {
				resized.release();
			}
		}
	}

	/**
	 * 取面积最大的人脸。
	 */
	private static FaceBox largest(List<FaceBox> faces) {
		FaceBox best = faces.get(0);
		for (int i = 1; i < faces.size(); i++) {
			if (area(faces.get(i)) > area(best)) {
				best = faces.get(i);
			}
		}
		return best;
	}

	private static float area(FaceBox box) {
		return (box.getX2() - box.getX1()) * (box.getY2() - box.getY1());
	}

	private static int[] box(FaceBox face) {
		if (face == null) {
			return null;
		}
		return new int[]{Math.round(face.getX1()), Math.round(face.getY1()),
			Math.round(face.getX2()), Math.round(face.getY2())};
	}

	/**
	 * 人脸框（浮点精度，供几何计算使用）。
	 */
	private static float[] faceBox(FaceBox face) {
		if (face == null) {
			return new float[]{0f, 0f, 0f, 0f};
		}
		return new float[]{face.getX1(), face.getY1(), face.getX2(), face.getY2()};
	}

	private static IdPhotoResult failure(boolean passed, String reason, String hint,
										 float qualityScore, Boolean qualityPassed,
										 float faceScore, int[] faceBox,
										 IdPhotoSpec spec, IdPhotoBackground background,
										 IdPhotoMaskSource maskSource, int[] crop) {
		return new IdPhotoResult(passed, reason, hint, null, 0, 0,
			qualityScore, qualityPassed, faceScore, faceBox,
			spec.getLabel(), background.getLabel(), maskSource.getLabel(), crop, false, 1);
	}

	private static double round(float value) {
		if (Float.isNaN(value) || Float.isInfinite(value)) {
			return 0d;
		}
		return Math.round(value * 1000d) / 1000d;
	}

}
