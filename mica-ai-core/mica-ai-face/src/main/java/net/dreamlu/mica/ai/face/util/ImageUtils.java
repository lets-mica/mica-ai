/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.util;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.model.FaceBox;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * 图像处理工具。
 */
public class ImageUtils {

	private ImageUtils() {
	}

	public static Mat byteArrayToMat(byte[] imageBytes) {
		if (imageBytes == null || imageBytes.length == 0) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.DETECTION_FAILED, "图像字节为空");
		}
		MatOfByte mob = new MatOfByte(imageBytes);
		Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
		if (mat.empty()) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.DETECTION_FAILED, "图像解码失败或格式不支持");
		}
		return mat;
	}

	public static Mat cropWithMargin(Mat image, FaceBox box, double scale, int outputSize) {
		if (image == null || image.empty() || box == null) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.LIVENESS_FAILED, "裁剪入参为空");
		}
		float cx = (box.getX1() + box.getX2()) / 2f;
		float cy = (box.getY1() + box.getY2()) / 2f;
		float halfW = (box.getX2() - box.getX1()) / 2f;
		float halfH = (box.getY2() - box.getY1()) / 2f;
		float half = Math.max(halfW, halfH) * (float) scale;

		int x1 = Math.max(0, (int) Math.floor(cx - half));
		int y1 = Math.max(0, (int) Math.floor(cy - half));
		int x2 = Math.min(image.cols(), (int) Math.ceil(cx + half));
		int y2 = Math.min(image.rows(), (int) Math.ceil(cy + half));

		int cw = x2 - x1;
		int ch = y2 - y1;
		if (cw <= 0 || ch <= 0) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.LIVENESS_FAILED, "外扩裁剪区域无效");
		}

		Rect roi = new Rect(x1, y1, cw, ch);
		Mat cropped = new Mat(image, roi);
		Mat resized = new Mat();
		try {
			Imgproc.resize(cropped, resized, new Size(outputSize, outputSize), 0, 0, Imgproc.INTER_LINEAR);
			return resized;
		} finally {
			cropped.release();
		}
	}

	public static float[] bgrHwcToChwFloat(Mat bgr, float scale, float offset) {
		int h = bgr.rows();
		int w = bgr.cols();
		int hw = h * w;
		byte[] pixels = new byte[hw * 3];
		bgr.get(0, 0, pixels);
		float[] data = new float[3 * hw];
		for (int i = 0; i < hw; i++) {
			int b = pixels[i * 3] & 0xFF;
			int g = pixels[i * 3 + 1] & 0xFF;
			int r = pixels[i * 3 + 2] & 0xFF;
			data[i] = (b - offset) * scale;
			data[hw + i] = (g - offset) * scale;
			data[2 * hw + i] = (r - offset) * scale;
		}
		return data;
	}

	public static float[] bgrHwcToRgbChwFloat(Mat bgr, float scale, float offset) {
		int h = bgr.rows();
		int w = bgr.cols();
		int hw = h * w;
		byte[] pixels = new byte[hw * 3];
		bgr.get(0, 0, pixels);
		float[] data = new float[3 * hw];
		for (int i = 0; i < hw; i++) {
			int b = pixels[i * 3] & 0xFF;
			int g = pixels[i * 3 + 1] & 0xFF;
			int r = pixels[i * 3 + 2] & 0xFF;
			data[i] = (r - offset) * scale;
			data[hw + i] = (g - offset) * scale;
			data[2 * hw + i] = (b - offset) * scale;
		}
		return data;
	}

	public static byte[] matToBytes(Mat bgr, String format, int quality) {
		if (bgr == null || bgr.empty()) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.ENCODE_FAILED, "待编码图像为空");
		}
		String ext = normalizeFormat(format);
		MatOfByte mob = new MatOfByte();
		MatOfInt params = new MatOfInt();
		try {
			if ("jpg".equals(ext) || "webp".equals(ext)) {
				int q = Math.max(1, Math.min(100, quality));
				params.fromArray(Imgcodecs.IMWRITE_JPEG_QUALITY, q);
			}
			if (!Imgcodecs.imencode("." + ext, bgr, mob, params)) {
				throw new MicaAiException(
					MicaAiException.ErrorCode.ENCODE_FAILED,
					"图片编码失败，格式: " + ext);
			}
			return mob.toArray();
		} finally {
			mob.release();
			params.release();
		}
	}

	private static String normalizeFormat(String format) {
		String f = format == null ? "" : format.trim().toLowerCase();
		if (f.startsWith(".")) {
			f = f.substring(1);
		}
		if ("jpeg".equals(f)) {
			f = "jpg";
		}
		switch (f) {
			case "png":
			case "jpg":
			case "bmp":
			case "webp":
				return f;
			default:
				throw new MicaAiException(
					MicaAiException.ErrorCode.ENCODE_FAILED,
					"不支持的图片格式: " + format + "（支持 png / jpg / bmp / webp）");
		}
	}

	public static double sharpness(Mat bgr) {
		if (bgr == null || bgr.empty()) {
			return 0d;
		}
		Mat gray = new Mat();
		Mat laplacian = new Mat();
		MatOfDouble mean = new MatOfDouble();
		MatOfDouble stddev = new MatOfDouble();
		try {
			Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
			Imgproc.Laplacian(gray, laplacian, CvType.CV_64F);
			Core.meanStdDev(laplacian, mean, stddev);
			double sd = stddev.get(0, 0)[0];
			return sd * sd;
		} finally {
			gray.release();
			laplacian.release();
			mean.release();
			stddev.release();
		}
	}

	public static void releaseAll(Mat... mats) {
		if (mats == null) {
			return;
		}
		for (Mat m : mats) {
			if (m != null) {
				m.release();
			}
		}
	}
}