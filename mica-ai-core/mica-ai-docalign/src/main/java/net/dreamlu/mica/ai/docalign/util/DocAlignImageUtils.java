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
package net.dreamlu.mica.ai.docalign.util;

import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.docalign.model.DocAlignResult;
import net.dreamlu.mica.ai.docalign.model.DocCorner;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * mica-ai-docalign 内部图像工具（CHW 张量化 / 热力图取质心 / 透视矫正）。
 *
 * <p>工具类，私有构造，线程安全（无状态）。
 */
public final class DocAlignImageUtils {

	private DocAlignImageUtils() {
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
	 * 把 BGR Mat 缩放到模型输入尺寸并转成 RGB CHW float（仅 {@code /255} 归一化）。
	 *
	 * <p>DocAligner 上游契约是纯 {@code img/255}，<b>没有</b> mean / std。
	 *
	 * @param bgr    原图 BGR Mat
	 * @param width  目标宽度
	 * @param height 目标高度
	 * @return RGB CHW 布局的 float 数组，长度 {@code 3 * width * height}
	 */
	public static float[] resizeToRgbChwFloat(Mat bgr, int width, int height) {
		Mat resized = new Mat();
		try {
			// Imgproc.resize 的 Size 是 (宽, 高)
			Imgproc.resize(bgr, resized, new Size(width, height), 0, 0, Imgproc.INTER_LINEAR);
			return bgrHwcToRgbChwFloat(resized);
		} finally {
			resized.release();
		}
	}

	/**
	 * 把 BGR HWC byte Mat 转成 RGB CHW float（值域 0~1）。
	 *
	 * @param bgr 待转换的 BGR Mat
	 * @return RGB CHW 布局的 float 数组
	 */
	public static float[] bgrHwcToRgbChwFloat(Mat bgr) {
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
			data[i] = r / 255f;
			data[hw + i] = g / 255f;
			data[2 * hw + i] = b / 255f;
		}
		return data;
	}

	/**
	 * 把单通道热力图缩放到原图尺寸（{@code INTER_LINEAR}）。
	 *
	 * @param heatmap 单通道 {@code CV_32FC1} 热力图
	 * @param width   目标宽度
	 * @param height  目标高度
	 * @return 缩放后的热力图，调用方负责 release
	 */
	public static Mat resizeHeatmap(Mat heatmap, int width, int height) {
		Mat out = new Mat();
		Imgproc.resize(heatmap, out, new Size(width, height), 0, 0, Imgproc.INTER_LINEAR);
		return out;
	}

	/**
	 * 从单通道热力图中定位角点：阈值二值化 → 取<b>面积最大</b>连通域 → 质心。
	 *
	 * <p>对齐上游 DocAligner 的 {@code postprocess}：热力图放大到原图尺寸后
	 * 按阈值二值化，取面积最大的轮廓质心。若阈值后没有任何前景像素，则回退到
	 * 热力图全局最大值位置，保证调用方始终拿到 4 个点。
	 *
	 * @param prob      单通道 {@code CV_32FC1} 热力图（0~1）
	 * @param threshold 二值化阈值（0~1）
	 * @return {@code [x, y, score]}；score 为该热力图通道的全局最大概率
	 */
	public static double[] locateCorner(Mat prob, float threshold) {
		Mat mask = new Mat();
		Mat binary = new Mat();
		Mat hierarchy = new Mat();
		List<MatOfPoint> contours = new ArrayList<>();
		try {
			Imgproc.threshold(prob, mask, threshold, 255, Imgproc.THRESH_BINARY);
			mask.convertTo(binary, CvType.CV_8UC1);
			Imgproc.findContours(binary, contours, hierarchy,
				Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

			double[] max = maxValue(prob);
			double score = max[0];
			double[] fallback = new double[]{max[1], max[2], score};
			if (contours.isEmpty()) {
				return fallback;
			}

			MatOfPoint best = null;
			double bestArea = -1d;
			for (MatOfPoint contour : contours) {
				double area = Math.abs(Imgproc.contourArea(contour));
				if (area > bestArea) {
					bestArea = area;
					best = contour;
				}
			}
			if (best == null) {
				return fallback;
			}
			// 多边形质心（鞋带公式），避免依赖 openpnp 绑定里缺失的 Moments 类
			Point[] points = best.toArray();
			double twiceArea = 0d;
			double cx = 0d;
			double cy = 0d;
			for (int i = 0; i < points.length; i++) {
				Point p = points[i];
				Point q = points[(i + 1) % points.length];
				double cross = p.x * q.y - q.x * p.y;
				twiceArea += cross;
				cx += (p.x + q.x) * cross;
				cy += (p.y + q.y) * cross;
			}
			double area = twiceArea * 0.5d;
			if (Math.abs(area) > 1e-6d) {
				return new double[]{cx / (3d * twiceArea), cy / (3d * twiceArea), score};
			}
			Rect rect = Imgproc.boundingRect(best);
			return new double[]{
				rect.x + rect.width / 2d,
				rect.y + rect.height / 2d,
				score};
		} finally {
			mask.release();
			binary.release();
			hierarchy.release();
			for (MatOfPoint contour : contours) {
				contour.release();
			}
		}
	}

	/**
	 * 扫描单通道 float Mat，返回 {@code [maxValue, maxX, maxY]}。
	 */
	private static double[] maxValue(Mat prob) {
		int w = prob.cols();
		int h = prob.rows();
		float[] data = new float[w * h];
		prob.get(0, 0, data);
		float best = -Float.MAX_VALUE;
		int bestIndex = 0;
		for (int i = 0; i < data.length; i++) {
			if (data[i] > best) {
				best = data[i];
				bestIndex = i;
			}
		}
		return new double[]{best, bestIndex % w, bestIndex / w};
	}

	/**
	 * 按角点边长推算透视矫正后的推荐输出尺寸。
	 *
	 * @param result 角点检测结果
	 * @return {@code [width, height]}，均为 {@code >= 1} 的整数
	 */
	public static int[] suggestOutputSize(DocAlignResult result) {
		DocCorner lt = result.getTopLeft();
		DocCorner rt = result.getTopRight();
		DocCorner rb = result.getBottomRight();
		DocCorner lb = result.getBottomLeft();
		double width = Math.max(lt.distanceTo(rt), lb.distanceTo(rb));
		double height = Math.max(lt.distanceTo(lb), rt.distanceTo(rb));
		return new int[]{
			Math.max(1, (int) Math.round(width)),
			Math.max(1, (int) Math.round(height))};
	}

	/**
	 * 按 4 个角点做透视矫正，把文档拉平成矩形。
	 *
	 * @param bgr      原图 BGR Mat
	 * @param result   角点检测结果（左上 / 右上 / 右下 / 左下）
	 * @param outWidth 输出宽度
	 * @param outHeight 输出高度
	 * @return 拉平后的 BGR Mat，调用方负责 release
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 输出尺寸非法
	 */
	public static Mat warp(Mat bgr, DocAlignResult result, int outWidth, int outHeight) {
		if (outWidth <= 0 || outHeight <= 0) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"输出宽高必须为正数，当前为 " + outWidth + "x" + outHeight);
		}
		Mat srcPoints = new Mat(4, 1, CvType.CV_32FC2);
		Mat dstPoints = new Mat(4, 1, CvType.CV_32FC2);
		Mat transform = null;
		Mat warped = new Mat();
		try {
			putPoint(srcPoints, 0, result.getTopLeft());
			putPoint(srcPoints, 1, result.getTopRight());
			putPoint(srcPoints, 2, result.getBottomRight());
			putPoint(srcPoints, 3, result.getBottomLeft());
			dstPoints.put(0, 0, new float[]{
				0f, 0f,
				outWidth, 0f,
				outWidth, outHeight,
				0f, outHeight});
			transform = Imgproc.getPerspectiveTransform(srcPoints, dstPoints);
			Imgproc.warpPerspective(bgr, warped, transform,
				new Size(outWidth, outHeight), Imgproc.INTER_LINEAR);
			return warped;
		} finally {
			srcPoints.release();
			dstPoints.release();
			if (transform != null) {
				transform.release();
			}
		}
	}

	private static void putPoint(Mat points, int index, DocCorner corner) {
		points.put(index, 0, new float[]{(float) corner.getX(), (float) corner.getY()});
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
}
