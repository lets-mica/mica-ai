/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.avatar;

import lombok.Getter;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.model.ModelManager;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Getter
public class AvatarExtractor {

	private static final int[] CARDINAL = {0, 90, 180, 270};
	private static final double ORIENT_MARGIN = 0.03;
	private static final double ORIENT_CROP_FACTOR = 1.6;
	private static final float ORIENT_THRESHOLD = 0.3f;

	private final FaceDetector detector;
	private final AvatarOptions defaults;

	public AvatarExtractor(ModelManager modelManager) {
		this(new FaceDetector(modelManager), AvatarOptions.defaults());
	}

	public AvatarExtractor(FaceDetector detector) {
		this(detector, AvatarOptions.defaults());
	}

	public AvatarExtractor(FaceDetector detector, AvatarOptions defaults) {
		this.detector = detector;
		this.defaults = defaults == null ? AvatarOptions.defaults() : defaults;
	}

	public AvatarResult extract(Mat image, AvatarOptions options) {
		AvatarOptions opts = options == null ? defaults : options;
		List<AvatarResult> results = extractAll(image, opts.toBuilder().maxFaces(1).build());
		if (results.isEmpty()) {
			throw new MicaAiException(ErrorCode.AVATAR_FAILED, "未检测到人脸");
		}
		return results.get(0);
	}

	public AvatarResult extract(Mat image) {
		return extract(image, defaults);
	}

	public List<AvatarResult> extractAll(Mat image, AvatarOptions options) {
		if (image == null || image.empty()) {
			throw new MicaAiException(ErrorCode.AVATAR_FAILED, "头像提取入参图像为空");
		}
		AvatarOptions opts = options == null ? defaults : options;
		opts.validate();

		Located located = locate(image, opts);
		List<FaceBox> boxes = located.boxes();
		if (boxes.isEmpty()) {
			return Collections.emptyList();
		}
		boxes.sort((a, b) -> Double.compare(area(b), area(a)));

		int limit = opts.getMaxFaces() > 0
			? Math.min(opts.getMaxFaces(), boxes.size()) : boxes.size();

		List<AvatarResult> results = new ArrayList<>(limit);
		try {
			for (int i = 0; i < limit; i++) {
				results.add(render(image, boxes.get(i), i, located.tiled(), opts));
			}
		} catch (RuntimeException e) {
			results.forEach(AvatarResult::release);
			throw e;
		}
		return results;
	}

	public List<AvatarResult> extractAll(Mat image) {
		return extractAll(image, defaults);
	}

	public static Mat drawDebug(Mat image, List<AvatarResult> results, AvatarOptions options) {
		if (image == null || image.empty()) {
			throw new MicaAiException(ErrorCode.AVATAR_FAILED, "标注入参图像为空");
		}
		Mat canvas = image.clone();
		if (results == null || results.isEmpty()) {
			return canvas;
		}
		for (AvatarResult r : results) {
			FaceBox b = r.getBox();
			Imgproc.rectangle(canvas, new Point(b.getX1(), b.getY1()),
				new Point(b.getX2(), b.getY2()), new Scalar(0, 0, 255), 2);
			float[][] q = r.getWindowQuad();
			if (q != null && q.length == 4) {
				Point[] pts = new Point[4];
				for (int i = 0; i < 4; i++) {
					pts[i] = new Point(q[i][0], q[i][1]);
				}
				Imgproc.polylines(canvas, Collections.singletonList(new MatOfPoint(pts)),
					true, new Scalar(0, 200, 0), 2);
			}
		}
		AvatarResult first = results.get(0);
		FaceBox b0 = first.getBox();
		Imgproc.putText(canvas,
			String.format("face %.0fpx sharp %.1f rot %d",
				first.getFaceSize(), first.getSharpness(), first.getOrientationDegrees()),
			new Point(b0.getX1(), Math.max(14, b0.getY1() - 6)),
			Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
		return canvas;
	}

	private Located locate(Mat image, AvatarOptions opts) {
		List<FaceBox> boxes = detector.detect(image);
		if (!boxes.isEmpty()) {
			return new Located(boxes, false);
		}
		if (!opts.isTileDetect()) {
			return new Located(Collections.emptyList(), false);
		}
		boxes = detector.detectTiled(image, opts.getTileSize(), opts.getTileOverlap(),
			(float) opts.getTileThreshold());
		return new Located(boxes, true);
	}

	private AvatarResult render(Mat image, FaceBox box, int index, boolean tiled, AvatarOptions opts) {
		int rotation = opts.isAutoOrient() ? chooseOrientation(image, box, opts) : 0;

		Mat rotated = null;
		try {
			Mat work = image;
			FaceBox workBox = box;
			if (rotation != 0) {
				rotated = rotateImage(image, rotation);
				work = rotated;
				workBox = rotateBox(box, rotation, image.cols(), image.rows());
			}

			Window window = windowAffine(workBox, opts);
			Mat avatar = renderWindow(work, window.affine(), window.side(),
				opts.getSize(), opts.backgroundScalar());

			AvatarResult result = new AvatarResult();
			result.setImage(avatar);
			result.setBox(box);
			result.setWindowQuad(rotateQuadBack(windowQuad(window), rotation,
				image.cols(), image.rows()));
			result.setSize(opts.getSize());
			result.setFaceSize(Math.max(box.getX2() - box.getX1(), box.getY2() - box.getY1()));
			result.setOrientationDegrees(rotation);
			result.setTiledDetection(tiled);
			result.setIndex(index);
			result.setSharpness(ImageUtils.sharpness(avatar));
			applyUsability(result, opts);
			return result;
		} finally {
			if (rotated != null) {
				rotated.release();
			}
		}
	}

	private int chooseOrientation(Mat image, FaceBox box, AvatarOptions opts) {
		double l = Math.max(box.getX2() - box.getX1(), box.getY2() - box.getY1());
		if (l <= 1) {
			return 0;
		}
		int w = image.cols();
		int h = image.rows();
		double half = l * ORIENT_CROP_FACTOR;
		double cx = (box.getX1() + box.getX2()) / 2.0;
		double cy = (box.getY1() + box.getY2()) / 2.0;
		int x1 = (int) Math.max(0, Math.floor(cx - half));
		int y1 = (int) Math.max(0, Math.floor(cy - half));
		int x2 = (int) Math.min(w, Math.ceil(cx + half));
		int y2 = (int) Math.min(h, Math.ceil(cy + half));
		int cw = x2 - x1;
		int ch = y2 - y1;
		if (cw < 16 || ch < 16) {
			return 0;
		}

		Mat crop = new Mat(image, new Rect(x1, y1, cw, ch));
		try {
			float baseScore = -1f;
			float bestScore = -1f;
			int bestDeg = 0;
			for (int deg : CARDINAL) {
				Mat candidate = null;
				try {
					candidate = deg == 0 ? crop : rotateImage(crop, deg);
					List<FaceBox> found = detector.detect(candidate, ORIENT_THRESHOLD);
					float top = found.isEmpty() ? -1f : found.get(0).getScore();
					if (deg == 0) {
						baseScore = top;
					}
					if (top > bestScore) {
						bestScore = top;
						bestDeg = deg;
					}
				} finally {
					if (candidate != null && candidate != crop) {
						candidate.release();
					}
				}
			}
			if (bestScore < 0f) {
				return 0;
			}
			if (bestDeg != 0 && bestScore - baseScore < ORIENT_MARGIN) {
				return 0;
			}
			return bestDeg;
		} finally {
			crop.release();
		}
	}

	static Mat rotateImage(Mat src, int degrees) {
		Mat dst = new Mat();
		switch (degrees) {
			case 0:
				src.copyTo(dst);
				return dst;
			case 90:
				Core.rotate(src, dst, Core.ROTATE_90_CLOCKWISE);
				return dst;
			case 180:
				Core.rotate(src, dst, Core.ROTATE_180);
				return dst;
			case 270:
				Core.rotate(src, dst, Core.ROTATE_90_COUNTERCLOCKWISE);
				return dst;
			default:
				throw new MicaAiException(
					ErrorCode.AVATAR_FAILED, "不支持的旋转角度: " + degrees);
		}
	}

	static float[] rotateForward(float x, float y, int degrees, int w, int h) {
		switch (degrees) {
			case 0:
				return new float[]{x, y};
			case 90:
				return new float[]{h - y, x};
			case 180:
				return new float[]{w - x, h - y};
			case 270:
				return new float[]{y, w - x};
			default:
				throw new MicaAiException(
					ErrorCode.AVATAR_FAILED, "不支持的旋转角度: " + degrees);
		}
	}

	static float[] rotateBack(float x, float y, int degrees, int w, int h) {
		switch (degrees) {
			case 0:
				return new float[]{x, y};
			case 90:
				return new float[]{y, h - x};
			case 180:
				return new float[]{w - x, h - y};
			case 270:
				return new float[]{w - y, x};
			default:
				throw new MicaAiException(
					ErrorCode.AVATAR_FAILED, "不支持的旋转角度: " + degrees);
		}
	}

	static FaceBox rotateBox(FaceBox box, int degrees, int w, int h) {
		if (degrees == 0) {
			return box;
		}
		float[] a = rotateForward(box.getX1(), box.getY1(), degrees, w, h);
		float[] b = rotateForward(box.getX2(), box.getY2(), degrees, w, h);
		float[][] src = box.getLandmarks();
		float[][] lm = null;
		if (src != null) {
			lm = new float[src.length][2];
			for (int i = 0; i < src.length; i++) {
				float[] p = rotateForward(src[i][0], src[i][1], degrees, w, h);
				lm[i][0] = p[0];
				lm[i][1] = p[1];
			}
		}
		return new FaceBox(Math.min(a[0], b[0]), Math.min(a[1], b[1]),
			Math.max(a[0], b[0]), Math.max(a[1], b[1]), box.getScore(), lm);
	}

	static float[][] rotateQuadBack(float[][] quad, int degrees, int w, int h) {
		if (degrees == 0 || quad == null) {
			return quad;
		}
		float[][] out = new float[quad.length][2];
		for (int i = 0; i < quad.length; i++) {
			float[] p = rotateBack(quad[i][0], quad[i][1], degrees, w, h);
			out[i][0] = p[0];
			out[i][1] = p[1];
		}
		return out;
	}

	private static void applyUsability(AvatarResult result, AvatarOptions opts) {
		int min = opts.getMinFaceSize();
		if (min > 0 && result.getFaceSize() < min) {
			result.setUsable(false);
			result.setUnusableReason(String.format(
				"人脸框长边 %.1fpx 低于 minFaceSize %d，输出为纯上采样", result.getFaceSize(), min));
		} else {
			result.setUsable(true);
			result.setUnusableReason(null);
		}
	}

	static final class Window {
		private final double[] affine;
		private final int side;

		Window(double[] affine, int side) {
			this.affine = affine;
			this.side = side;
		}

		public double[] affine() {
			return affine;
		}

		public int side() {
			return side;
		}
	}

	private static final class Located {
		private final List<FaceBox> boxes;
		private final boolean tiled;

		private Located(List<FaceBox> boxes, boolean tiled) {
			this.boxes = boxes;
			this.tiled = tiled;
		}

		public List<FaceBox> boxes() {
			return boxes;
		}

		public boolean tiled() {
			return tiled;
		}
	}

	static Window windowAffine(FaceBox box, AvatarOptions opts) {
		float x1 = box.getX1();
		float y1 = box.getY1();
		float x2 = box.getX2();
		float y2 = box.getY2();
		double bw = x2 - x1;
		double bh = y2 - y1;
		if (bw <= 0 || bh <= 0) {
			throw new MicaAiException(
				ErrorCode.AVATAR_FAILED,
				String.format("人脸框尺寸无效: %.1f × %.1f", bw, bh));
		}
		double l = Math.max(bw, bh);

		double sideF = l * opts.getFaceScale();
		int side = (int) Math.max(2, Math.min(16384, Math.round(sideF)));
		double k = side / sideF;

		double cx = (x1 + x2) / 2.0;
		double cy = (y1 + y2) / 2.0 + opts.getVerticalOffset() * bh;

		double cos = 1.0;
		double sin = 0.0;
		double theta = Math.toRadians(opts.getRotationDegrees());
		if (opts.isDeRotate()) {
			float[][] lm = box.getLandmarks();
			if (lm != null && lm.length >= 2) {
				double detected = Math.atan2(lm[1][1] - lm[0][1], lm[1][0] - lm[0][0]);
				if (Double.isFinite(detected)) {
					theta += detected;
				}
			}
		}
		if (theta != 0.0) {
			cos = Math.cos(theta);
			sin = Math.sin(theta);
		}

		double m00 = k * cos;
		double m01 = k * sin;
		double m10 = -k * sin;
		double m11 = k * cos;
		double half = side / 2.0;
		double tx = half - (m00 * cx + m01 * cy);
		double ty = half - (m10 * cx + m11 * cy);
		return new Window(new double[]{m00, m01, tx, m10, m11, ty}, side);
	}

	static Mat renderWindow(Mat image, double[] affine, int side, int size, Scalar background) {
		Mat m = new Mat(2, 3, CvType.CV_64F);
		Mat window = new Mat();
		try {
			m.put(0, 0, affine);
			Imgproc.warpAffine(image, window, m, new Size(side, side),
				Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, background);
			if (side == size) {
				Mat out = window;
				window = null;
				return out;
			}
			Mat out = new Mat();
			int interpolation = side > size ? Imgproc.INTER_AREA : Imgproc.INTER_CUBIC;
			Imgproc.resize(window, out, new Size(size, size), 0, 0, interpolation);
			return out;
		} finally {
			m.release();
			if (window != null) {
				window.release();
			}
		}
	}

	static float[][] windowQuad(Window window) {
		double[] a = window.affine();
		double det = a[0] * a[4] - a[1] * a[3];
		if (Math.abs(det) < 1e-12) {
			throw new MicaAiException(
				ErrorCode.AVATAR_FAILED, "裁剪仿射矩阵不可逆");
		}
		double ia = a[4] / det;
		double ib = -a[1] / det;
		double ic = -a[3] / det;
		double id = a[0] / det;
		double itx = -(ia * a[2] + ib * a[5]);
		double ity = -(ic * a[2] + id * a[5]);

		double s = window.side();
		double[][] corners = {{0, 0}, {s, 0}, {s, s}, {0, s}};
		float[][] quad = new float[4][2];
		for (int i = 0; i < 4; i++) {
			quad[i][0] = (float) (ia * corners[i][0] + ib * corners[i][1] + itx);
			quad[i][1] = (float) (ic * corners[i][0] + id * corners[i][1] + ity);
		}
		return quad;
	}

	private static double area(FaceBox box) {
		return (box.getX2() - box.getX1()) * (box.getY2() - box.getY1());
	}
}