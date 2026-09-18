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
package net.dreamlu.mica.ai.layout.util;

import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * mica-ai-layout 内部图像工具，零 Spring、零 face 依赖。
 */
public final class LayoutImageUtils {

	private LayoutImageUtils() {
	}

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
	 * 把 BGR {@link Mat} 转成 RGB + CHW float，并按 (x - mean) / std 归一化。
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