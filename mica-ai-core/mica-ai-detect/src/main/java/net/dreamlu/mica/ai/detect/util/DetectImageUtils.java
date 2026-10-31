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
package net.dreamlu.mica.ai.detect.util;

import lombok.RequiredArgsConstructor;
import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * mica-ai-detect 内部图像工具（letterbox / CHW 张量化），零 Spring 依赖。
 *
 * <p>工具类，私有构造，线程安全（无状态）。
 */
@UtilityClass
public class DetectImageUtils {

	/**
	 * letterbox 结果：模型输入图 + 反算回原图所需的缩放 / 偏移。
	 */
	@RequiredArgsConstructor
	public static final class Letterbox {

		/**
		 * 模型输入图（RGB、方形、{@code CV_8UC3}），调用方负责 release
		 */
		public final Mat image;

		/**
		 * 缩放比：{@code 模型坐标 = 原图坐标 * ratio + 偏移}
		 */
		public final float ratio;

		/**
		 * 顶部填充像素
		 */
		public final int top;

		/**
		 * 左侧填充像素
		 */
		public final int left;

	}

	/**
	 * 把图像字节数组解码为 BGR {@link Mat}。
	 *
	 * @param imageBytes 图像文件字节（png / jpg 等）
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
	 * 等比缩放并居中填充到方形，再 BGR→RGB。
	 *
	 * <p>与 opencv_zoo 参考实现一致：{@code ratio = min(size/w, size/h)}，
	 * 缩放后的边长按截断取整，余下部分以 {@code padValue} 居中填充。
	 *
	 * @param bgr           原图 BGR Mat
	 * @param size          目标方形边长
	 * @param padValue      填充灰度值（NanoDet 用 0，YOLOX 用 114）
	 * @param interpolation OpenCV 插值方式
	 * @return letterbox 结果，内部的 {@link Letterbox#image} 由调用方 release
	 */
	public static Letterbox letterbox(Mat bgr, int size, int padValue, int interpolation) {
		int w = bgr.cols();
		int h = bgr.rows();
		float ratio = Math.min((float) size / w, (float) size / h);
		int newW = Math.max(1, Math.min(size, (int) (w * ratio)));
		int newH = Math.max(1, Math.min(size, (int) (h * ratio)));
		int top = (size - newH) / 2;
		int left = (size - newW) / 2;

		Mat resized = new Mat();
		Mat padded = new Mat();
		Mat rgb = new Mat();
		try {
			Imgproc.resize(bgr, resized, new Size(newW, newH), 0, 0, interpolation);
			Core.copyMakeBorder(resized, padded, top, size - newH - top, left,
				size - newW - left, Core.BORDER_CONSTANT,
				new Scalar(padValue, padValue, padValue));
			Imgproc.cvtColor(padded, rgb, Imgproc.COLOR_BGR2RGB);
			return new Letterbox(rgb, ratio, top, left);
		} finally {
			resized.release();
			padded.release();
		}
	}

	/**
	 * 把 RGB HWC byte Mat 转成 CHW float，并按 {@code (x - mean) / std} 归一化。
	 *
	 * <p>{@code mean}/{@code std} 为 {@code null} 时不归一化，直接输出 0~255 原始值
	 * （YOLOX 参考实现即如此）。注意 <b>不做</b> {@code /255}：NanoDet 的官方统计量
	 * 本身就是 0~255 量纲（{@code mean=[103.53,116.28,123.675]}）。
	 *
	 * @param rgb  RGB {@code CV_8UC3} Mat
	 * @param mean 每通道均值（RGB 顺序），可为 null
	 * @param std  每通道标准差（RGB 顺序），可为 null
	 * @return CHW 布局的 float 数组，长度 {@code 3 * h * w}
	 */
	public static float[] toChwFloat(Mat rgb, float[] mean, float[] std) {
		int h = rgb.rows();
		int w = rgb.cols();
		int hw = h * w;
		byte[] pixels = new byte[hw * 3];
		rgb.get(0, 0, pixels);
		float[] data = new float[3 * hw];
		float[] m = normalizeParams(mean, 0f);
		float[] s = normalizeParams(std, 1f);
		for (int i = 0; i < hw; i++) {
			for (int c = 0; c < 3; c++) {
				float v = pixels[i * 3 + c] & 0xFF;
				// NanoDet 的 mean/std 按 BGR 顺序给出却应用于 RGB 图（上游实现即如此），
				// 这里原样复刻，保证与参考实现的数值一致
				data[c * hw + i] = (v - m[c]) / s[c];
			}
		}
		return data;
	}

	private static float[] normalizeParams(float[] params, float defaultValue) {
		if (params == null) {
			return new float[]{defaultValue, defaultValue, defaultValue};
		}
		return params;
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
