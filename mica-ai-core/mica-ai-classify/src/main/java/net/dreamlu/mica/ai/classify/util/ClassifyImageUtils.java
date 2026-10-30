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
package net.dreamlu.mica.ai.classify.util;

import net.dreamlu.mica.ai.classify.config.ClassifyChannelOrder;
import net.dreamlu.mica.ai.classify.config.ClassifyInterpolation;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * mica-ai-classify 内部图像工具，零 Spring。
 *
 * <p>工具类，私有构造，线程安全（无状态）。
 */
public final class ClassifyImageUtils {

	private ClassifyImageUtils() {
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
			throw new MicaAiException(ErrorCode.INFERENCE_FAILED, "图像字节为空");
		}
		MatOfByte mob = new MatOfByte(imageBytes);
		try {
			Mat mat = Imgcodecs.imdecode(mob, Imgcodecs.IMREAD_COLOR);
			if (mat.empty()) {
				throw new MicaAiException(ErrorCode.INFERENCE_FAILED,
					"图像解码失败或格式不支持");
			}
			return mat;
		} finally {
			mob.release();
		}
	}

	/**
	 * 按 PaddleClas 推理流水线把图像转成归一化的 CHW float 数组。
	 *
	 * <p>两种预处理契约：
	 * <ul>
	 *   <li>{@code resizeShort > 0}：短边缩放到 {@code resizeShort}，再<b>中心裁剪</b>
	 *       {@code inputWidth×inputHeight}（PULC 官方 Eval/Infer 默认，256 → 224）</li>
	 *   <li>{@code resizeShort == 0}：直接 resize 到 {@code inputWidth×inputHeight}
	 *       （{@code table_attribute} 的官方配置）</li>
	 * </ul>
	 * ⚠️ {@link Imgproc#resize} 的 {@link Size} 是 {@code (宽, 高)}，不要写反。
	 *
	 * @param bgr          原图（BGR）
	 * @param inputWidth   模型输入宽度（NCHW 最后一维）
	 * @param inputHeight  模型输入高度
	 * @param resizeShort  短边缩放目标；0 表示直接 resize
	 * @param mean         3 通道均值
	 * @param std          3 通道标准差
	 * @param channelOrder 通道顺序
	 * @param interpolation 插值方式
	 * @return CHW 布局的归一化 float 数组，长度 {@code 3 * inputWidth * inputHeight}
	 */
	public static float[] toChwFloat(Mat bgr, int inputWidth, int inputHeight, int resizeShort,
									 float[] mean, float[] std,
									 ClassifyChannelOrder channelOrder,
									 ClassifyInterpolation interpolation) {
		Mat resized = new Mat();
		Mat cropped = null;
		try {
			if (resizeShort > 0) {
				int rows = bgr.rows();
				int cols = bgr.cols();
				double scale = (double) resizeShort / Math.min(rows, cols);
				int newWidth = Math.max(resizeShort, (int) Math.round(cols * scale));
				int newHeight = Math.max(resizeShort, (int) Math.round(rows * scale));
				Imgproc.resize(bgr, resized, new Size(newWidth, newHeight),
					0, 0, toCvInterpolation(interpolation));
				int x = Math.max(0, (newWidth - inputWidth) / 2);
				int y = Math.max(0, (newHeight - inputHeight) / 2);
				Mat roi = new Mat(resized, new Rect(x, y, inputWidth, inputHeight));
				cropped = roi.clone();
				roi.release();
			} else {
				Imgproc.resize(bgr, resized, new Size(inputWidth, inputHeight),
					0, 0, toCvInterpolation(interpolation));
				cropped = resized;
			}
			return hwcToChwFloat(cropped, mean, std, channelOrder);
		} finally {
			if (cropped != resized) {
				releaseAll(cropped);
			}
			releaseAll(resized);
		}
	}

	/**
	 * 把 HWC byte Mat 转成归一化的 CHW float 数组。
	 *
	 * @param mat          待转换的 3 通道 Mat
	 * @param mean         3 通道均值
	 * @param std          3 通道标准差
	 * @param channelOrder {@link ClassifyChannelOrder#BGR} 表示直接使用 OpenCV 原生 BGR，
	 *                     {@link ClassifyChannelOrder#RGB} 表示把 BGR 翻转为 RGB
	 * @return CHW 布局的归一化 float 数组
	 */
	public static float[] hwcToChwFloat(Mat mat, float[] mean, float[] std,
										ClassifyChannelOrder channelOrder) {
		int h = mat.rows();
		int w = mat.cols();
		int hw = h * w;
		byte[] pixels = new byte[hw * 3];
		mat.get(0, 0, pixels);
		boolean toRgb = channelOrder == ClassifyChannelOrder.RGB;
		float[] data = new float[3 * hw];
		for (int i = 0; i < hw; i++) {
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

	private static int toCvInterpolation(ClassifyInterpolation interpolation) {
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
