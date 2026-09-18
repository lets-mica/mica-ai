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
package net.dreamlu.mica.ai.face.util;

import net.dreamlu.mica.ai.common.exception.ErrorCode;
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

	/**
	 * 解码任意常见格式（jpg / png / bmp / webp 等）的字节数组为 BGR {@link Mat}。
	 *
	 * @param imageBytes 图像字节（非空）
	 * @return BGR Mat（3 通道），调用方负责 {@code release()}
	 * @throws MicaAiException {@link ErrorCode#DETECTION_FAILED} 输入为空或解码失败
	 */
	public static Mat byteArrayToMat(byte[] imageBytes) {
		if (imageBytes == null || imageBytes.length == 0) {
			throw new MicaAiException(
				ErrorCode.DETECTION_FAILED, "图像字节为空");
		}
		MatOfByte mob = new MatOfByte(imageBytes);
		Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
		if (mat.empty()) {
			throw new MicaAiException(
				ErrorCode.DETECTION_FAILED, "图像解码失败或格式不支持");
		}
		return mat;
	}

	/**
	 * 按人脸框外扩指定比例裁剪，并缩放到指定输出尺寸。
	 *
	 * @param image      输入图像
	 * @param box        人脸框
	 * @param scale      外扩比例，相对人脸框长边
	 * @param outputSize 输出正方形边长（像素）
	 * @return 裁剪并缩放后的图像
	 * @throws MicaAiException 入参为空或裁剪区域无效时抛出，错误码 {@link ErrorCode#LIVENESS_FAILED}
	 */
	public static Mat cropWithMargin(Mat image, FaceBox box, double scale, int outputSize) {
		if (image == null || image.empty() || box == null) {
			throw new MicaAiException(
				ErrorCode.LIVENESS_FAILED, "裁剪入参为空");
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
				ErrorCode.LIVENESS_FAILED, "外扩裁剪区域无效");
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

	/**
	 * 把 BGR {@link Mat} 转成 CHW float，并做 scale / offset 线性变换。
	 *
	 * @param bgr    原图 BGR Mat
	 * @param scale  像素缩放系数
	 * @param offset 像素偏移量
	 * @return CHW 布局的 float 数组
	 */
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

	/**
	 * 将 BGR 图像转为 RGB、CHW 排布的 float 数组，用于模型输入。
	 *
	 * @param bgr    BGR 输入图像
	 * @param scale  像素值缩放系数
	 * @param offset 像素值偏移量
	 * @return RGB CHW 排布的 float 数组
	 */
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

	/**
	 * 将 BGR 图像编码为指定格式的字节数组。
	 *
	 * @param bgr     BGR 输入图像
	 * @param format  图片格式，支持 png / jpg / bmp / webp
	 * @param quality JPEG 压缩质量，取值 [0, 100]，其它格式忽略
	 * @return 编码后的字节数组
	 * @throws MicaAiException 图像为空或格式不支持时抛出，错误码 {@link ErrorCode#ENCODE_FAILED}
	 */
	public static byte[] matToBytes(Mat bgr, String format, int quality) {
		if (bgr == null || bgr.empty()) {
			throw new MicaAiException(
				ErrorCode.ENCODE_FAILED, "待编码图像为空");
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
					ErrorCode.ENCODE_FAILED,
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
					ErrorCode.ENCODE_FAILED,
					"不支持的图片格式: " + format + "（支持 png / jpg / bmp / webp）");
		}
	}

	/**
	 * 计算图像清晰度（拉普拉斯算子标准差），值越大越清晰。
	 *
	 * @param bgr BGR 输入图像
	 * @return 清晰度得分，入参图像为空时返回 0
	 */
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

	/**
	 * 批量安全 release：null 容忍，逐个 release。
	 *
	 * @param mats 任意数量 Mat（可为 null / 单个 / 数组）
	 */
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