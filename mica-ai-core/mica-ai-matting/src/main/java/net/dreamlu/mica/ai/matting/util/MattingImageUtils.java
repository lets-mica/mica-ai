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
package net.dreamlu.mica.ai.matting.util;

import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.matting.config.MattingInterpolation;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * mica-ai-matting 内部图像工具，零 Spring、零 face 依赖。
 *
 * <p>工具类，私有构造，线程安全（无状态）。
 */
public final class MattingImageUtils {

	private MattingImageUtils() {
	}

	/**
	 * 把图像字节数组解码为 BGR {@link Mat}。
	 *
	 * @param imageBytes 图像文件字节（png / jpg 等 OpenCV 支持的格式）
	 * @return 解码后的 BGR Mat，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 字节为空或解码失败
	 */
	public static Mat byteArrayToMat(byte[] imageBytes) {
		if (imageBytes == null || imageBytes.length == 0) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "图像字节为空");
		}
		MatOfByte mob = new MatOfByte(imageBytes);
		try {
			Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
			if (mat.empty()) {
				throw new MicaAiException(
					ErrorCode.INFERENCE_FAILED, "图像解码失败或格式不支持");
			}
			return mat;
		} finally {
			mob.release();
		}
	}

	/**
	 * 把 BGR {@link Mat} 等比缩放到模型输入尺寸，并转成 RGB CHW float，按 (x/255 - mean) / std 归一化。
	 *
	 * <p>u2netp 是 320×320 固定输入，官方参考实现直接 resize（不 letterbox）。
	 *
	 * @param bgr       原图 BGR Mat
	 * @param inputSize 目标方形边长
	 * @param mean      每通道均值（<b>RGB</b> 顺序）
	 * @param std       每通道标准差（<b>RGB</b> 顺序）
	 * @return RGB CHW 布局的归一化 float 数组，长度 {@code 3 * inputSize * inputSize}
	 */
	public static float[] resizeToRgbChwFloat(Mat bgr, int inputSize, float[] mean, float[] std) {
		Mat resized = new Mat();
		try {
			Imgproc.resize(bgr, resized, new Size(inputSize, inputSize));
			return bgrHwcToRgbChwFloat(resized, mean, std);
		} finally {
			resized.release();
		}
	}

	/**
	 * 把 BGR HWC byte Mat 转成 RGB CHW float，并按 (x/255 - mean) / std 归一化。
	 *
	 * @param bgr  待转换的 BGR Mat
	 * @param mean 每通道均值（RGB 顺序）
	 * @param std  每通道标准差（RGB 顺序）
	 * @return RGB CHW 布局的归一化 float 数组
	 */
	public static float[] bgrHwcToRgbChwFloat(Mat bgr, float[] mean, float[] std) {
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
			data[i] = ((r / 255f) - mean[0]) / std[0];
			data[hw + i] = ((g / 255f) - mean[1]) / std[1];
			data[2 * hw + i] = ((b / 255f) - mean[2]) / std[2];
		}
		return data;
	}

	/**
	 * 把 320×320 的 float 掩码缩放到目标尺寸。
	 *
	 * @param mask  单通道 {@code CV_32FC1} 掩码
	 * @param width 目标宽度
	 * @param height 目标高度
	 * @param interpolation 插值方式
	 * @return 缩放后的单通道 {@code CV_32FC1} 掩码，调用方负责 release
	 */
	public static Mat resizeMask(Mat mask, int width, int height, MattingInterpolation interpolation) {
		Mat out = new Mat();
		Imgproc.resize(mask, out, new Size(width, height), 0, 0, toCvInterpolation(interpolation));
		return out;
	}

	/**
	 * 掩码 min-max 拉伸到 [0,1]；极差过小时返回全 0 掩码（避免除零把噪声放大成全白）。
	 *
	 * @param mask 原地归一化的单通道 float 掩码
	 */
	public static void minMaxNormalize(Mat mask) {
		Core.MinMaxLocResult mm = Core.minMaxLoc(mask);
		double range = mm.maxVal - mm.minVal;
		if (range < 1e-8) {
			mask.setTo(new Scalar(0));
			return;
		}
		double scale = 1d / range;
		// Core.subtract(Scalar, Mat, Mat) 在 openpnp 绑定里不存在，需显式构造同形 Scalar Mat
		Mat minMat = new Mat(mask.size(), mask.type(), new Scalar(mm.minVal));
		try {
			Core.subtract(mask, minMat, mask);
			Core.multiply(mask, new Scalar(scale), mask);
		} finally {
			minMat.release();
		}
	}

	/**
	 * 按阈值把 float 掩码二值化为 {@code 0/255} 的 {@code CV_8UC1}。
	 *
	 * @param alpha     单通道 {@code CV_32FC1} 掩码
	 * @param threshold 二值化阈值
	 * @return 二值掩码，调用方负责 release
	 */
	public static Mat threshold(Mat alpha, float threshold) {
		Mat out = new Mat();
		Imgproc.threshold(alpha, out, threshold, 255, Imgproc.THRESH_BINARY);
		out.convertTo(out, CvType.CV_8UC1);
		return out;
	}

	/**
	 * 用 alpha 把原图合成到纯色底上。
	 *
	 * @param bgr    原图 BGR Mat
	 * @param alpha  与原图同尺寸的单通道 {@code CV_32FC1} 掩码
	 * @param bgRgb  底色（RGB，0~255）
	 * @return 合成后的 BGR Mat，调用方负责 release
	 */
	public static Mat compositeOnColor(Mat bgr, Mat alpha, int[] bgRgb) {
		Mat src = new Mat();
		Mat alpha3 = new Mat();
		Mat alphaInv = new Mat();
		Mat ones = new Mat();
		Mat fg = new Mat();
		Mat bg = new Mat();
		Mat sum = new Mat();
		Mat out = new Mat();
		try {
			bgr.convertTo(src, CvType.CV_32FC3);
			Imgproc.cvtColor(alpha, alpha3, Imgproc.COLOR_GRAY2BGR);
			// openpnp 绑定没有 Core.subtract(Scalar, Mat, Mat)，用 Mat 形式的 1 代替
			ones = ones3(bgr.size());
			Core.subtract(ones, alpha3, alphaInv);
			// Scalar 是 (B, G, R, A) 顺序，配置给的是 RGB
			Core.multiply(src, alpha3, fg);
			bg = new Mat(bgr.size(), CvType.CV_32FC3,
				new Scalar(bgRgb[2], bgRgb[1], bgRgb[0]));
			Core.multiply(bg, alphaInv, bg);
			Core.add(fg, bg, sum);
			sum.convertTo(out, CvType.CV_8UC3);
			return out;
		} finally {
			src.release();
			alpha3.release();
			alphaInv.release();
			ones.release();
			fg.release();
			bg.release();
			sum.release();
		}
	}

	/**
	 * 用 alpha 把原图转成带透明通道的 BGRA。
	 *
	 * @param bgr   原图 BGR Mat
	 * @param alpha 与原图同尺寸的单通道 {@code CV_32FC1} 掩码
	 * @return {@code CV_8UC4} 的 BGRA Mat，调用方负责 release
	 */
	public static Mat toBgra(Mat bgr, Mat alpha) {
		Mat alpha8 = new Mat();
		Mat bgra = new Mat();
		Mat[] channels = null;
		try {
			alpha.convertTo(alpha8, CvType.CV_8UC1, 255d);
			List<Mat> list = new ArrayList<>(4);
			Core.split(bgr, list);
			channels = list.toArray(new Mat[0]);
			List<Mat> merged = new ArrayList<>(4);
			merged.add(channels[0]);
			merged.add(channels[1]);
			merged.add(channels[2]);
			merged.add(alpha8);
			Core.merge(merged, bgra);
			return bgra;
		} finally {
			alpha8.release();
			if (channels != null) {
				for (Mat c : channels) {
					c.release();
				}
			}
		}
	}

	/**
	 * 把 {@link Mat} 编码为 PNG 字节数组（透明底走 PNG，避免 JPEG 丢弃 alpha）。
	 *
	 * @param mat 待编码的 Mat（BGR / BGRA / 单通道均可）
	 * @return PNG 字节数组
	 * @throws MicaAiException {@link ErrorCode#ENCODE_FAILED} 编码失败
	 */
	public static byte[] imencodePng(Mat mat) {
		MatOfByte mob = new MatOfByte();
		try {
			if (!Imgcodecs.imencode(".png", mat, mob)) {
				throw new MicaAiException(ErrorCode.ENCODE_FAILED, "PNG 编码失败");
			}
			return mob.toArray();
		} finally {
			mob.release();
		}
	}

	/**
	 * 把 {@link Mat} 编码为 JPEG 字节数组。
	 *
	 * @param mat     待编码的 Mat（必须为 1 或 3 通道）
	 * @param quality 质量 1~100
	 * @return JPEG 字节数组
	 * @throws MicaAiException {@link ErrorCode#ENCODE_FAILED} 编码失败
	 */
	public static byte[] imencodeJpeg(Mat mat, int quality) {
		MatOfByte mob = new MatOfByte();
		Mat bgr8 = new Mat();
		try {
			if (mat.channels() == 4) {
				Imgproc.cvtColor(mat, bgr8, Imgproc.COLOR_BGRA2BGR);
			} else if (mat.channels() == 1) {
				Imgproc.cvtColor(mat, bgr8, Imgproc.COLOR_GRAY2BGR);
			} else {
				mat.convertTo(bgr8, CvType.CV_8UC3);
			}
			Imgcodecs.imencode(".jpg", bgr8, mob,
				new MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality));
			return mob.toArray();
		} finally {
			bgr8.release();
			mob.release();
		}
	}

	/**
	 * 构造全 1 的三通道 {@code CV_32FC3} Mat。
	 *
	 * <p>openpnp 的 OpenCV 绑定没有 {@code Core.subtract(Scalar, Mat, Mat)} 重载，
	 * 需要 {@code 1 - alpha} 时必须显式给出 Mat 形式的标量，否则编译不通过。
	 *
	 * @param size 目标尺寸
	 * @return 全 1 的 {@code CV_32FC3} Mat，调用方负责 release
	 */
	public static Mat ones3(Size size) {
		return new Mat(size, CvType.CV_32FC3, new Scalar(1, 1, 1));
	}

	private static int toCvInterpolation(MattingInterpolation interpolation) {		if (interpolation == null) {
			return Imgproc.INTER_LINEAR;
		}
		switch (interpolation) {
			case NEAREST:
				return Imgproc.INTER_NEAREST;
			case CUBIC:
				return Imgproc.INTER_CUBIC;
			case LINEAR:
			default:
				return Imgproc.INTER_LINEAR;
		}
	}

	/**
	 * 批量释放 Mat，null 元素自动跳过。
	 *
	 * @param mats 待释放的 Mat 可变参数
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
