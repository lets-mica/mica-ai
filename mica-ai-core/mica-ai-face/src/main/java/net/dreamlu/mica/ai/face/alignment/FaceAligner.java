/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.alignment;

import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.model.FaceBox;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * 人脸对齐器。
 *
 * <p>基于 5 个关键点估计相似变换矩阵，将人脸区域 warp 到 112×112 的标准正脸。
 * 参考点坐标取自 InsightFace 标准 ArcFace 对齐模板。
 */
public class FaceAligner {

	public static final int OUTPUT_SIZE = 112;

	private static final float[][] REFERENCE_112 = {
		{38.2946f, 51.6963f},
		{73.5318f, 51.5014f},
		{56.0252f, 71.7366f},
		{41.5493f, 92.3655f},
		{70.7299f, 92.2041f}
	};

	public Mat align(Mat image, FaceBox faceBox) {
		if (image == null || image.empty() || faceBox == null || faceBox.getLandmarks() == null) {
			throw new MicaAiException(
				ErrorCode.ALIGNMENT_FAILED, "对齐入参为空");
		}
		float[][] landmarks = faceBox.getLandmarks();
		if (landmarks.length != 5) {
			throw new MicaAiException(
				ErrorCode.ALIGNMENT_FAILED, "需要 5 个关键点，实际 " + landmarks.length);
		}

		MatOfPoint2f src = new MatOfPoint2f();
		MatOfPoint2f dst = new MatOfPoint2f();
		Point[] srcPoints = new Point[5];
		Point[] dstPoints = new Point[5];
		for (int i = 0; i < 5; i++) {
			srcPoints[i] = new Point(landmarks[i][0], landmarks[i][1]);
			dstPoints[i] = new Point(REFERENCE_112[i][0], REFERENCE_112[i][1]);
		}
		src.fromArray(srcPoints);
		dst.fromArray(dstPoints);

		Mat m = null;
		Mat aligned = new Mat();
		try {
			m = Calib3d.estimateAffinePartial2D(src, dst);
			if (m.empty()) {
				throw new MicaAiException(ErrorCode.ALIGNMENT_FAILED, "仿射矩阵估计失败");
			}
			Imgproc.warpAffine(image, aligned, m, new Size(OUTPUT_SIZE, OUTPUT_SIZE));
		} finally {
			src.release();
			dst.release();
			if (m != null) {
				m.release();
			}
		}
		return aligned;
	}
}