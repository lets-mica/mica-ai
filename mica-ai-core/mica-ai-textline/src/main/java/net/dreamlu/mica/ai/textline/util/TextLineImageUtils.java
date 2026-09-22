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
package net.dreamlu.mica.ai.textline.util;

import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.textline.config.TextLineChannelOrder;
import net.dreamlu.mica.ai.textline.config.TextLineInterpolation;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * mica-ai-textline 内部图像工具，零 Spring、零 face 依赖。
 *
 * <p>工具类，私有构造，线程安全（无状态）。
 */
public final class TextLineImageUtils {

	private TextLineImageUtils() {
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
	 * 把文本行图缩放到模型输入尺寸，按 {@code (x/255 - mean) / std} 归一化，输出 CHW float。
	 *
	 * <p>⚠️ <b>尺寸语义</b>：{@code Imgproc.resize} 的 {@link Size} 是
	 * {@code (宽, 高)}；模型输入是 {@code [1,3,80,160]}（H=80, W=160），
	 * 因此必须传 {@code new Size(inputWidth, inputHeight)}。写反会让文本行被拉成
	 * 「竖排」，方向判定随之失真。
	 *
	 * @param bgr          文本行图（BGR）
	 * @param inputWidth   目标宽度
	 * @param inputHeight  目标高度
	 * @param mean         3 通道均值
	 * @param std          3 通道标准差
	 * @param channelOrder 通道顺序
	 * @param interpolation 插值方式
	 * @return CHW 布局的归一化 float 数组，长度 {@code 3 * inputWidth * inputHeight}
	 */
	public static float[] resizeToChwFloat(Mat bgr, int inputWidth, int inputHeight,
										   float[] mean, float[] std,
										   TextLineChannelOrder channelOrder,
										   TextLineInterpolation interpolation) {
		Mat resized = new Mat();
		try {
			Imgproc.resize(bgr, resized, new Size(inputWidth, inputHeight),
				0, 0, toCvInterpolation(interpolation));
			return hwcToChwFloat(resized, mean, std, channelOrder);
		} finally {
			resized.release();
		}
	}

	/**
	 * 把 HWC byte Mat 转成归一化的 CHW float 数组。
	 *
	 * @param mat          待转换的 3 通道 Mat
	 * @param mean         3 通道均值
	 * @param std          3 通道标准差
	 * @param channelOrder {@link TextLineChannelOrder#BGR} 表示输入本身就是 BGR，
	 *                     {@link TextLineChannelOrder#RGB} 表示需要把通道翻转为 RGB
	 * @return CHW 布局的归一化 float 数组
	 */
	public static float[] hwcToChwFloat(Mat mat, float[] mean, float[] std,
										TextLineChannelOrder channelOrder) {
		int h = mat.rows();
		int w = mat.cols();
		int hw = h * w;
		byte[] pixels = new byte[hw * 3];
		mat.get(0, 0, pixels);
		boolean toRgb = channelOrder == TextLineChannelOrder.RGB;
		float[] data = new float[3 * hw];
		for (int i = 0; i < hw; i++) {
			// OpenCV 原生是 BGR；ch0=B ch1=G ch2=R
			int b = pixels[i * 3] & 0xFF;
			int g = pixels[i * 3 + 1] & 0xFF;
			int r = pixels[i * 3 + 2] & 0xFF;
			float c0 = toRgb ? r : b;
			float c1 = g;
			float c2 = toRgb ? b : r;
			data[i] = ((c0 / 255f) - mean[0]) / std[0];
			data[hw + i] = ((c1 / 255f) - mean[1]) / std[1];
			data[2 * hw + i] = ((c2 / 255f) - mean[2]) / std[2];
		}
		return data;
	}

	/**
	 * 把文本行旋转 180 度（转正）。
	 *
	 * @param bgr 原文本行图
	 * @return 旋转后的新 Mat，调用方负责 release
	 */
	public static Mat rotate180(Mat bgr) {
		Mat out = new Mat();
		org.opencv.core.Core.rotate(bgr, out, org.opencv.core.Core.ROTATE_180);
		return out;
	}

	/**
	 * 把 {@link Mat} 编码为 PNG 字节数组。
	 *
	 * @param mat 待编码的 Mat
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

	private static int toCvInterpolation(TextLineInterpolation interpolation) {
		if (interpolation == null) {
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
}
