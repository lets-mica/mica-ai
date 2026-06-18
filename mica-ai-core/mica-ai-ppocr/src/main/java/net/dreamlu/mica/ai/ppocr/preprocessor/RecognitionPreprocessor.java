package net.dreamlu.mica.ai.ppocr.preprocessor;

import lombok.ToString;
import net.dreamlu.mica.ai.ppocr.utils.NdArrayUtils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.List;

/**
 * 识别模型预处理：OCRResizeNormImg + ToBatch。
 */
@ToString
public final class RecognitionPreprocessor {
	public static final int CHANNELS = 3;
	public static final int HEIGHT = 48;
	public static final int W_MIN = 320;
	public static final int W_MAX = 3200;
	public static final int[] DEFAULT_SHAPE = {CHANNELS, HEIGHT, W_MIN};

	private final int h;
	private final int wMin;
	private final int wMax;

	public RecognitionPreprocessor() {
		this(HEIGHT, W_MIN, W_MAX);
	}

	public RecognitionPreprocessor(int h, int wMin, int wMax) {
		if (h <= 0) {
			throw new IllegalArgumentException("h must be > 0");
		}
		if (wMin <= 0 || wMax <= 0 || wMax < wMin) {
			throw new IllegalArgumentException("invalid wMin/wMax: " + wMin + "/" + wMax);
		}
		this.h = h;
		this.wMin = wMin;
		this.wMax = wMax;
	}

	public Result call(List<Mat> imgs) {
		int n = imgs.size();
		if (n == 0) {
			return new Result(new float[0], new int[]{0, CHANNELS, h, 0});
		}

		List<float[][]> perImgChw = new java.util.ArrayList<>(n);
		int[] widths = new int[n];
		int[] actualWs = new int[n];
		for (int i = 0; i < n; i++) {
			Mat img = imgs.get(i);
			int srcH = img.rows();
			int srcW = img.cols();
			double whRatio = Math.max((double) wMin / h, (double) srcW / srcH);
			int targetW = (int) (h * whRatio);

			int actualW;
			Mat resized;
			if (targetW > wMax) {
				resized = new Mat();
				Imgproc.resize(img, resized, new Size(wMax, h), 0, 0, Imgproc.INTER_LINEAR);
				actualW = wMax;
				targetW = wMax;
			} else {
				actualW = Math.min(NdArrayUtils.ceilDiv(h * srcW, srcH), targetW);
				resized = new Mat();
				Imgproc.resize(img, resized, new Size(actualW, h), 0, 0, Imgproc.INTER_LINEAR);
			}
			widths[i] = targetW;
			actualWs[i] = actualW;

			Mat f = NdArrayUtils.toFloat32(resized);
			float[] hwcRaw = NdArrayUtils.matToFlatHwc(f);
			f.release();
			int nPix = hwcRaw.length;
			for (int k = 0; k < nPix; k++) {
				hwcRaw[k] = (hwcRaw[k] / 255.0f - 0.5f) / 0.5f;
			}
			float[] hwcFlat = hwcRaw;
			float[][] chw = new float[CHANNELS][h * actualW];
			int hw = h * actualW;
			for (int j = 0; j < hw; j++) {
				int baseHwc = j * CHANNELS;
				chw[0][j] = hwcFlat[baseHwc];
				chw[1][j] = hwcFlat[baseHwc + 1];
				chw[2][j] = hwcFlat[baseHwc + 2];
			}
			perImgChw.add(chw);
		}

		int maxW = 0;
		for (int w : widths) {
			if (w > maxW) maxW = w;
		}
		int totalSize = n * CHANNELS * h * maxW;
		float[] data = new float[totalSize];
		int chwSize = CHANNELS * h * maxW;
		for (int i = 0; i < n; i++) {
			int actualW = actualWs[i];
			int destBase = i * chwSize;
			float[][] chw = perImgChw.get(i);
			for (int c = 0; c < CHANNELS; c++) {
				float[] chwC = chw[c];
				int cOffset = destBase + c * h * maxW;
				int hw = h * actualW;
				for (int j = 0; j < hw; j++) {
					int hh = j / actualW;
					int ww = j % actualW;
					data[cOffset + hh * maxW + ww] = chwC[j];
				}
			}
		}

		return new Result(data, new int[]{n, CHANNELS, h, maxW});
	}

	public record Result(float[] data, int[] shape) {}
}
