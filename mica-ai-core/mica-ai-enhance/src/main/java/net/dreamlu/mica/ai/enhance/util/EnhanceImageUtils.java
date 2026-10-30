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
package net.dreamlu.mica.ai.enhance.util;

import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * 超分图像工具：BGR / RGB 与 CHW float 张量互转、tile 裁剪、PNG 编码。
 *
 * <p>归一化契约：输入只做 {@code /255}（<b>无 mean / std</b>），输出 {@code ×255} 后 clip 到
 * {@code [0,255]}。
 */
public final class EnhanceImageUtils {

	private EnhanceImageUtils() {
	}

	/**
	 * 解码图像字节为 BGR Mat。
	 *
	 * @param imageBytes 图像字节
	 * @return BGR Mat；入参为空或解码失败时抛异常
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 解码失败
	 */
	public static Mat byteArrayToMat(byte[] imageBytes) {
		if (imageBytes == null || imageBytes.length == 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "图像字节为空");
		}
		MatOfByte mob = new MatOfByte(imageBytes);
		Mat mat = null;
		try {
			mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
			if (mat.empty()) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "图像解码失败");
			}
			return mat;
		} finally {
			mob.release();
		}
	}

	/**
	 * BGR Mat 转 RGB CHW float 数组（值域 {@code [0,1]}）。
	 *
	 * @param bgr BGR 三通道 Mat
	 * @return 长度为 {@code 3 * 宽 * 高} 的 CHW 数组（R 平面 → G 平面 → B 平面）
	 */
	public static float[] bgrToRgbChwFloat(Mat bgr) {
		int width = bgr.cols();
		int height = bgr.rows();
		int plane = width * height;
		byte[] rgbData = new byte[plane * 3];
		Mat rgb = new Mat();
		try {
			Imgproc.cvtColor(bgr, rgb, Imgproc.COLOR_BGR2RGB);
			rgb.get(0, 0, rgbData);
		} finally {
			rgb.release();
		}
		float[] chw = new float[plane * 3];
		for (int i = 0; i < plane; i++) {
			// byte 为有符号类型，必须 & 0xff 后再归一化，否则 >127 的像素会变成负数
			chw[i] = (rgbData[i * 3] & 0xff) / 255f;
			chw[plane + i] = (rgbData[i * 3 + 1] & 0xff) / 255f;
			chw[2 * plane + i] = (rgbData[i * 3 + 2] & 0xff) / 255f;
		}
		return chw;
	}

	/**
	 * CHW float 数组（RGB，值域约 {@code [0,1]}）转 BGR 三通道 Mat。
	 *
	 * @param chw    CHW 数组，长度必须为 {@code 3 * width * height}
	 * @param width  图像宽度
	 * @param height 图像高度
	 * @return BGR 三通道 {@code CV_8UC3} Mat，值 clip 到 {@code [0,255]}
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 数组长度不符
	 */
	public static Mat chwFloatToBgrMat(float[] chw, int width, int height) {
		if (chw == null || width <= 0 || height <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT, "CHW 参数非法");
		}
		int plane = width * height;
		if (chw.length != plane * 3) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"CHW 数组长度应为 " + (plane * 3) + "，实际 " + chw.length);
		}
		byte[] rgbData = new byte[plane * 3];
		for (int i = 0; i < plane; i++) {
			rgbData[i * 3] = (byte) clamp255(chw[i]);
			rgbData[i * 3 + 1] = (byte) clamp255(chw[plane + i]);
			rgbData[i * 3 + 2] = (byte) clamp255(chw[2 * plane + i]);
		}
		Mat rgb = new Mat(height, width, CvType.CV_8UC3);
		Mat bgr = new Mat();
		try {
			rgb.put(0, 0, rgbData);
			Imgproc.cvtColor(rgb, bgr, Imgproc.COLOR_RGB2BGR);
			return bgr;
		} finally {
			rgb.release();
		}
	}

	/**
	 * 裁剪 ROI 并复制为独立 Mat（原 Mat 释放后仍可用）。
	 *
	 * @param src    源 Mat
	 * @param x      左上角 x
	 * @param y      左上角 y
	 * @param width  宽
	 * @param height 高
	 * @return 独立的子图 Mat
	 */
	public static Mat crop(Mat src, int x, int y, int width, int height) {
		return new Mat(src, new Rect(x, y, width, height)).clone();
	}

	/**
	 * BGR Mat 编码为 PNG 字节。
	 *
	 * @param mat BGR Mat
	 * @return PNG 字节
	 * @throws MicaAiException {@link ErrorCode#ENCODE_FAILED} 编码失败
	 */
	public static byte[] imencodePng(Mat mat) {
		MatOfByte out = new MatOfByte();
		try {
			if (!Imgcodecs.imencode(".png", mat, out)) {
				throw new MicaAiException(ErrorCode.ENCODE_FAILED, "PNG 编码失败");
			}
			return out.toArray();
		} finally {
			out.release();
		}
	}

	/**
	 * 批量释放 Mat，忽略 null。
	 *
	 * @param mats 待释放的 Mat
	 */
	public static void releaseAll(Mat... mats) {
		if (mats == null) {
			return;
		}
		for (Mat mat : mats) {
			if (mat != null) {
				mat.release();
			}
		}
	}

	private static int clamp255(float value) {
		float scaled = value * 255f;
		if (scaled < 0f) {
			return 0;
		}
		if (scaled > 255f) {
			return 255;
		}
		return (int) Math.round(scaled);
	}
}
