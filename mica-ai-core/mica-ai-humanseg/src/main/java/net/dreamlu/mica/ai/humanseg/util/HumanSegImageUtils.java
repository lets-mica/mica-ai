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
package net.dreamlu.mica.ai.humanseg.util;

import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * mica-ai-humanseg 内部图像工具（resize / CHW 张量化 / 掩码合成），零 Spring 依赖。
 *
 * <p>工具类，私有构造，线程安全（无状态）。
 */
@UtilityClass
public class HumanSegImageUtils {

	/**
	 * 把图像字节数组解码为 BGR {@link Mat}。
	 *
	 * @param imageBytes 图像文件字节（png / jpg 等 OpenCV 支持的格式）
	 * @return 解码后的 BGR Mat，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#INFERENCE_FAILED} 字节为空或解码失败
	 */
	public static Mat byteArrayToMat(byte[] imageBytes) {
		if (imageBytes == null || imageBytes.length == 0) {
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "图像字节为空");
		}
		MatOfByte mob = new MatOfByte(imageBytes);
		try {
			Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
			if (mat.empty()) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "图像解码失败或格式不支持");
			}
			return mat;
		} finally {
			mob.release();
		}
	}

	/**
	 * 把 BGR {@link Mat} 缩放到模型输入尺寸，转成 RGB CHW float，按 {@code (x/255 - mean) / std} 归一化。
	 *
	 * <p>PP-HumanSeg 官方参考实现直接 resize（不 letterbox、不保持宽高比），
	 * 掩码随后会 resize 回原尺寸。
	 *
	 * @param bgr    原图 BGR Mat
	 * @param width  目标宽度
	 * @param height 目标高度
	 * @param mean   每通道均值（<b>RGB</b> 顺序）
	 * @param std    每通道标准差（<b>RGB</b> 顺序）
	 * @return RGB CHW 布局的归一化 float 数组，长度 {@code 3 * width * height}
	 */
	public static float[] resizeToRgbChwFloat(Mat bgr, int width, int height,
											  float[] mean, float[] std) {
		Mat resized = new Mat();
		try {
			// Imgproc.resize 的 Size 是 (宽, 高)
			Imgproc.resize(bgr, resized, new Size(width, height));
			return bgrHwcToRgbChwFloat(resized, mean, std);
		} finally {
			resized.release();
		}
	}

	/**
	 * 把 BGR HWC byte Mat 转成 RGB CHW float，并按 {@code (x/255 - mean) / std} 归一化。
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
	 * 把模型输出的掩码缩放到原图尺寸。
	 *
	 * @param mask   单通道 {@code CV_32FC1} 掩码
	 * @param width  目标宽度
	 * @param height 目标高度
	 * @return 缩放后的单通道 {@code CV_32FC1} 掩码，调用方负责 release
	 */
	public static Mat resizeMask(Mat mask, int width, int height) {
		Mat out = new Mat();
		// Size 是 (宽, 高)，写反会把宽高互换
		Imgproc.resize(mask, out, new Size(width, height), 0, 0, Imgproc.INTER_LINEAR);
		return out;
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
			ones = new Mat(bgr.size(), CvType.CV_32FC3, new Scalar(1, 1, 1));
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
