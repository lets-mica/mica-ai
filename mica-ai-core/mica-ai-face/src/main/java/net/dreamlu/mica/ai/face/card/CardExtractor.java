/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.card;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.AccessLevel;
import lombok.experimental.Accessors;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.face.detection.FaceDetector;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 证件卡片提取器：从复杂背景的照片中定位身份证 / 护照 / 银行卡等标准卡片，
 * 做透视矫正并输出尺寸固定、方向摆正的卡片图。
 */
@Slf4j
@Getter
public class CardExtractor {

	private static final double[] APPROX_EPSILONS = {0.015, 0.02, 0.03, 0.05};
	private static final int OPEN_KERNEL_SIZE = 5;
	private static final double AREA_WEIGHT = 0.6;
	private static final float ORIENT_THRESHOLD = 0.30f;
	private static final double ORIENT_MARGIN = 0.03;
	private static final int CLAHE_GRID = 8;
	private static final double DUPLICATE_IOU = 0.5;

	private final FaceDetector detector;
	private final CardOptions defaults;

	public CardExtractor() {
		this(null, CardOptions.defaults());
	}

	public CardExtractor(CardOptions defaults) {
		this(null, defaults);
	}

	public CardExtractor(FaceDetector detector) {
		this(detector, CardOptions.defaults());
	}

	public CardExtractor(FaceDetector detector, CardOptions defaults) {
		this.detector = detector;
		CardOptions opts = defaults == null ? CardOptions.defaults() : defaults;
		this.defaults = detector == null && opts.isAutoOrient()
			? opts.toBuilder().autoOrient(false).build() : opts;
	}

	public CardResult extract(Mat image, CardOptions options) {
		List<CardResult> results = extractAll(image, options, 1);
		if (results.isEmpty()) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED, "未在图中找到证件卡片");
		}
		return results.get(0);
	}

	public CardResult extract(Mat image) {
		return extract(image, defaults);
	}

	public CardResult extract(byte[] imageBytes, CardOptions options) {
		Mat image = ImageUtils.byteArrayToMat(imageBytes);
		try {
			return extract(image, options);
		} finally {
			image.release();
		}
	}

	public CardResult extract(byte[] imageBytes) {
		return extract(imageBytes, defaults);
	}

	public List<CardResult> extractAll(Mat image, CardOptions options) {
		return extractAll(image, options, Integer.MAX_VALUE);
	}

	public List<CardResult> extractAll(Mat image) {
		return extractAll(image, defaults);
	}

	private List<CardResult> extractAll(Mat image, CardOptions options, int limit) {
		if (image == null || image.empty()) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED, "卡片提取入参图像为空");
		}
		CardOptions opts = options == null ? defaults : options;
		opts.validate();

		List<Quad> quads = locate(image, opts);
		if (quads.isEmpty()) {
			return Collections.emptyList();
		}
		int count = Math.min(limit, quads.size());
		List<CardResult> results = new ArrayList<>(count);
		try {
			for (int i = 0; i < count; i++) {
				results.add(render(image, quads.get(i), i, opts));
			}
		} catch (RuntimeException e) {
			results.forEach(CardResult::release);
			throw e;
		}
		return results;
	}

	public static Mat drawDebug(Mat image, List<CardResult> results) {
		if (image == null || image.empty()) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED, "标注入参图像为空");
		}
		Mat canvas = image.clone();
		if (results == null || results.isEmpty()) {
			return canvas;
		}
		Scalar[] cornerColors = {
			new Scalar(0, 0, 255),
			new Scalar(0, 200, 0),
			new Scalar(255, 0, 0),
			new Scalar(0, 215, 255)};
		for (CardResult r : results) {
			Point[] pts = toPoints(r.getQuad());
			MatOfPoint poly = new MatOfPoint(pts);
			try {
				Imgproc.polylines(canvas, Collections.singletonList(poly),
					true, new Scalar(0, 0, 255), 2);
			} finally {
				poly.release();
			}
			for (int k = 0; k < pts.length; k++) {
				Imgproc.circle(canvas, pts[k], 5, cornerColors[k], -1);
			}
			Point anchor = new Point(
				pts[0].x, Math.max(16, pts[0].y - 8));
			Imgproc.putText(canvas, String.format("card %.0fpx ratio %.3f score %.2f rot %d",
					r.getCardSize(), r.getAspectRatio(), r.getScore(), r.getRotationDegrees()),
				anchor, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(0, 0, 255), 1, Imgproc.LINE_AA);
		}
		return canvas;
	}

	public static Mat drawDebug(Mat image, CardResult result) {
		return drawDebug(image, result == null ? Collections.emptyList()
			: Collections.singletonList(result));
	}

	private List<Quad> locate(Mat image, CardOptions opts) {
		int w = image.cols();
		int h = image.rows();
		double factor = resizeFactor(w, h, opts.getDetectMaxSide());

		Mat scaled = null;
		Mat mask = null;
		Mat hierarchy = new Mat();
		List<MatOfPoint> contours = new ArrayList<>();
		try {
			Mat work = image;
			if (factor < 1.0) {
				scaled = new Mat();
				Imgproc.resize(image, scaled,
					new Size(Math.max(1, (int) Math.round(w * factor)),
					Math.max(1, (int) Math.round(h * factor))),
					0, 0, Imgproc.INTER_AREA);
				work = scaled;
			}
			int ww = work.cols();
			int wh = work.rows();

			mask = buildMask(work, opts);
			Imgproc.findContours(mask, contours, hierarchy,
				Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

			double expected = opts.expectedAspectRatio();
			double tolerance = opts.getAspectTolerance();
			double minArea = opts.getMinAreaRatio() * ww * (double) wh;
			double invFactor = 1.0 / factor;

			List<Quad> candidates = new ArrayList<>();
			for (MatOfPoint contour : contours) {
				double area = Imgproc.contourArea(contour);
				if (area < minArea) {
					continue;
				}
				float[][] quad = approxQuad(contour);
				if (quad == null) {
					continue;
				}
				if (touchesBorder(quad, ww, wh, opts.getBorderMargin())) {
					continue;
				}
				double ratio = measuredRatio(quad);
				if (ratio <= 0) {
					continue;
				}
				double deviation = Math.abs(ratio - expected) / expected;
				double skew = edgePairDeviation(quad);
				if (deviation > tolerance || skew > tolerance) {
					continue;
				}
				candidates.add(new Quad(
					scaleQuad(quad, invFactor),
					area * invFactor * invFactor,
					ratio,
					1.0 - deviation / tolerance,
					0.0));
			}
			return rank(candidates);
		} finally {
			contours.forEach(Mat::release);
			hierarchy.release();
			if (mask != null) {
				mask.release();
			}
			if (scaled != null) {
				scaled.release();
			}
		}
	}

	private static Mat buildMask(Mat bgr, CardOptions opts) {
		Mat hsv = new Mat();
		Mat v = new Mat();
		Mat s = new Mat();
		Mat vMask = new Mat();
		Mat sMask = new Mat();
		Mat satMask = new Mat();
		try {
			Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV);
			Core.extractChannel(hsv, v, 2);
			Core.extractChannel(hsv, s, 1);

			Imgproc.threshold(v, vMask, 0, 255,
				Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
			Imgproc.threshold(s, sMask, 0, 255,
				Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);
			Imgproc.threshold(s, satMask, opts.getSaturationCeiling(), 255,
				Imgproc.THRESH_BINARY_INV);

			Core.bitwise_and(vMask, sMask, vMask);
			Core.bitwise_and(vMask, satMask, vMask);

			int closeSize = opts.getCloseKernelSize();
			Mat close = Imgproc.getStructuringElement(
				Imgproc.MORPH_RECT, new Size(closeSize, closeSize));
			try {
				Imgproc.morphologyEx(vMask, vMask, Imgproc.MORPH_CLOSE, close);
				Imgproc.morphologyEx(vMask, vMask, Imgproc.MORPH_CLOSE, close);
			} finally {
				close.release();
			}

			Mat open = Imgproc.getStructuringElement(
				Imgproc.MORPH_RECT, new Size(OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE));
			try {
				Imgproc.morphologyEx(vMask, vMask, Imgproc.MORPH_OPEN, open);
			} finally {
				open.release();
			}
			return vMask;
		} finally {
			hsv.release();
			v.release();
			s.release();
			sMask.release();
			satMask.release();
		}
	}

	static float[][] approxQuad(MatOfPoint contour) {
		Point[] pts = contour.toArray();
		if (pts.length < 4) {
			return null;
		}
		MatOfPoint2f curve = new MatOfPoint2f(pts);
		try {
			double perimeter = Imgproc.arcLength(curve, true);
			if (!(perimeter > 0)) {
				return null;
			}
			for (double epsilon : APPROX_EPSILONS) {
				MatOfPoint2f approx = new MatOfPoint2f();
				try {
					Imgproc.approxPolyDP(curve, approx, epsilon * perimeter, true);
					Point[] ap = approx.toArray();
					if (ap.length == 4 && isConvex(ap)) {
						return toQuad(ap);
					}
				} finally {
					approx.release();
				}
			}
			return null;
		} finally {
			curve.release();
		}
	}

	static boolean isConvex(Point[] p) {
		if (Math.abs(shoelaceArea(p)) < 1.0) {
			return false;
		}
		int sign = 0;
		for (int i = 0; i < 4; i++) {
			Point a = p[i];
			Point b = p[(i + 1) % 4];
			Point c = p[(i + 2) % 4];
			double cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
			if (Math.abs(cross) < 1e-9) {
				continue;
			}
			int s = cross > 0 ? 1 : -1;
			if (sign == 0) {
				sign = s;
			} else if (s != sign) {
				return false;
			}
		}
		return sign != 0;
	}

	private static double shoelaceArea(Point[] p) {
		double sum = 0;
		for (int i = 0; i < 4; i++) {
			Point a = p[i];
			Point b = p[(i + 1) % 4];
			sum += a.x * b.y - b.x * a.y;
		}
		return sum / 2.0;
	}

	static boolean touchesBorder(float[][] quad, int width, int height, int margin) {
		for (float[] p : quad) {
			if (p[0] <= margin || p[1] <= margin
				|| p[0] >= width - 1 - margin || p[1] >= height - 1 - margin) {
				return true;
			}
		}
		return false;
	}

	static double measuredRatio(float[][] quad) {
		double e0 = distance(quad[0], quad[1]);
		double e1 = distance(quad[1], quad[2]);
		double e2 = distance(quad[2], quad[3]);
		double e3 = distance(quad[3], quad[0]);
		double w = (e0 + e2) / 2.0;
		double h = (e1 + e3) / 2.0;
		if (w <= 0 || h <= 0) {
			return 0;
		}
		return Math.max(w, h) / Math.min(w, h);
	}

	static double edgePairDeviation(float[][] quad) {
		double e0 = distance(quad[0], quad[1]);
		double e1 = distance(quad[1], quad[2]);
		double e2 = distance(quad[2], quad[3]);
		double e3 = distance(quad[3], quad[0]);
		double m0 = Math.max(e0, e2);
		double m1 = Math.max(e1, e3);
		double d0 = m0 <= 0 ? 1.0 : Math.abs(e0 - e2) / m0;
		double d1 = m1 <= 0 ? 1.0 : Math.abs(e1 - e3) / m1;
		return Math.max(d0, d1);
	}

	static float[][] orderQuad(float[][] quad) {
		Point[] p = toPoints(quad);
		double cx = (p[0].x + p[1].x + p[2].x + p[3].x) / 4.0;
		double cy = (p[0].y + p[1].y + p[2].y + p[3].y) / 4.0;
		List<Integer> order = new ArrayList<>(Arrays.asList(0, 1, 2, 3));
		order.sort((a, b) -> Double.compare(
			Math.atan2(p[a].y - cy, p[a].x - cx),
			Math.atan2(p[b].y - cy, p[b].x - cx)));

		int start = 0;
		double minSum = Double.MAX_VALUE;
		for (int i = 0; i < 4; i++) {
			double sum = p[order.get(i)].x + p[order.get(i)].y;
			if (sum < minSum) {
				minSum = sum;
				start = i;
			}
		}
		Point[] seq = new Point[4];
		for (int i = 0; i < 4; i++) {
			seq[i] = p[order.get((start + i) % 4)];
		}
		if (distance(seq[0], seq[1]) < distance(seq[0], seq[3])) {
			seq = new Point[]{seq[1], seq[2], seq[3], seq[0]};
		}
		return toQuad(seq);
	}

	private static List<Quad> rank(List<Quad> candidates) {
		if (candidates.isEmpty()) {
			return Collections.emptyList();
		}
		double maxArea = 0;
		for (Quad q : candidates) {
			maxArea = Math.max(maxArea, q.area());
		}
		List<Quad> scored = new ArrayList<>(candidates.size());
		for (Quad q : candidates) {
			double areaTerm = maxArea <= 0 ? 1.0 : q.area() / maxArea;
			double score = q.ratioScore() * ((1 - AREA_WEIGHT) + AREA_WEIGHT * areaTerm);
			scored.add(new Quad(q.points(), q.area(), q.ratio(), q.ratioScore(), score));
		}
		scored.sort((a, b) -> Double.compare(b.score(), a.score()));

		List<Quad> kept = new ArrayList<>(scored.size());
		for (Quad q : scored) {
			boolean duplicate = false;
			for (Quad k : kept) {
				if (quadIoU(q.points(), k.points()) > DUPLICATE_IOU) {
					duplicate = true;
					break;
				}
			}
			if (!duplicate) {
				kept.add(q);
			}
		}
		return kept;
	}

	static double quadIoU(float[][] a, float[][] b) {
		MatOfPoint2f ma = new MatOfPoint2f(toPoints(a));
		MatOfPoint2f mb = new MatOfPoint2f(toPoints(b));
		Mat intersection = new Mat();
		try {
			double inter = Imgproc.intersectConvexConvex(ma, mb, intersection);
			if (inter <= 0) {
				return 0;
			}
			double areaA = Math.abs(Imgproc.contourArea(ma));
			double areaB = Math.abs(Imgproc.contourArea(mb));
			double union = areaA + areaB - inter;
			return union <= 0 ? 0 : inter / union;
		} finally {
			ma.release();
			mb.release();
			intersection.release();
		}
	}

	private CardResult render(Mat image, Quad quad, int index, CardOptions opts) {
		float[][] ordered = orderQuad(quad.points());
		Mat card = warp(image, ordered, opts);
		try {
			Orientation orientation = resolveOrientation(card, opts);
			if (orientation.degrees() != 0) {
				Mat rotated = new Mat();
				try {
					Core.rotate(card, rotated, rotateCode(orientation.degrees()));
				} catch (RuntimeException e) {
					rotated.release();
					throw e;
				}
				card.release();
				card = rotated;
			}

			Mat output = enhance(card, opts);
			card.release();
			card = null;

			CardResult result = new CardResult();
			result.setImage(output);
			result.setQuad(ordered);
			result.setAspectRatio(quad.ratio());
			result.setScore(quad.score());
			result.setRotationDegrees(orientation.degrees());
			result.setAutoOriented(orientation.auto());
			result.setCardSize(longEdge(ordered));
			result.setIndex(index);
			result.setSharpness(ImageUtils.sharpness(output));
			applyUsability(result, opts);
			return result;
		} finally {
			if (card != null) {
				card.release();
			}
		}
	}

	private Orientation resolveOrientation(Mat card, CardOptions opts) {
		int manual = ((opts.getRotationDegrees() % 360) + 360) % 360;
		if (manual != 0) {
			return new Orientation(manual, false);
		}
		if (!opts.isAutoOrient() || detector == null) {
			return new Orientation(0, false);
		}
		return new Orientation(decideOrientation(card), true);
	}

	private int decideOrientation(Mat card) {
		double upright = maxFaceScore(card);
		Mat flipped = new Mat();
		try {
			Core.rotate(card, flipped, Core.ROTATE_180);
			double upsideDown = maxFaceScore(flipped);
			return upsideDown > upright + ORIENT_MARGIN ? 180 : 0;
		} finally {
			flipped.release();
		}
	}

	private double maxFaceScore(Mat image) {
		if (detector == null) {
			return 0;
		}
		double best = 0;
		for (FaceBox box : detector.detect(image, ORIENT_THRESHOLD)) {
			best = Math.max(best, box.getScore());
		}
		return best;
	}

	private static Mat warp(Mat image, float[][] quad, CardOptions opts) {
		int outW = opts.getOutputWidth();
		int outH = opts.getOutputHeight();
		Mat src = new Mat(4, 1, CvType.CV_32FC2);
		Mat dst = new Mat(4, 1, CvType.CV_32FC2);
		Mat transform = null;
		try {
			src.put(0, 0, flatten(quad));
			dst.put(0, 0, new double[]{
				0, 0,
				outW - 1, 0,
				outW - 1, outH - 1,
				0, outH - 1});
			transform = Imgproc.getPerspectiveTransform(src, dst);
			Mat out = new Mat();
			Imgproc.warpPerspective(image, out, transform, new Size(outW, outH),
				Imgproc.INTER_LANCZOS4, Core.BORDER_REPLICATE, new Scalar(0));
			return out;
		} finally {
			src.release();
			dst.release();
			if (transform != null) {
				transform.release();
			}
		}
	}

	private static Mat enhance(Mat card, CardOptions opts) {
		if (!opts.isEnhance()) {
			return card.clone();
		}
		Mat blur = new Mat();
		Mat sharp = new Mat();
		try {
			int k = sharpenKernel(opts.getSharpenSigma());
			double sigma = opts.getSharpenSigma();
			Imgproc.GaussianBlur(card, blur, new Size(k, k), sigma, sigma);
			Core.addWeighted(card, opts.getSharpenAmount(), blur,
				1.0 - opts.getSharpenAmount(), 0, sharp);
			if (opts.getClaheClip() <= 0) {
				Mat out = sharp;
				sharp = null;
				return out;
			}
			return applyClahe(sharp, opts);
		} finally {
			blur.release();
			if (sharp != null) {
				sharp.release();
			}
		}
	}

	private static Mat applyClahe(Mat sharp, CardOptions opts) {
		Mat lab = new Mat();
		Mat l = new Mat();
		Mat enhancedL = new Mat();
		try {
			Imgproc.cvtColor(sharp, lab, Imgproc.COLOR_BGR2Lab);
			Core.extractChannel(lab, l, 0);
			CLAHE clahe = Imgproc.createCLAHE(opts.getClaheClip(),
				new Size(CLAHE_GRID, CLAHE_GRID));
			clahe.apply(l, enhancedL);
			Core.insertChannel(enhancedL, lab, 0);
			Mat out = new Mat();
			Imgproc.cvtColor(lab, out, Imgproc.COLOR_Lab2BGR);
			return out;
		} finally {
			lab.release();
			l.release();
			enhancedL.release();
		}
	}

	static int sharpenKernel(double sigma) {
		int k = 2 * (int) Math.ceil(sigma * 3.0) + 1;
		return Math.max(3, k % 2 == 1 ? k : k + 1);
	}

	private static void applyUsability(CardResult result, CardOptions opts) {
		int min = opts.getMinCardSize();
		if (min > 0 && result.getCardSize() < min) {
			result.setUsable(false);
			result.setUnusableReason(String.format(
				"卡片长边 %.0fpx 低于 minCardSize %d，输出为 %.1f 倍上采样，细节不足以支撑文字识别",
				result.getCardSize(), min, opts.getOutputWidth() / result.getCardSize()));
		} else {
			result.setUsable(true);
			result.setUnusableReason(null);
		}
	}

	private static double resizeFactor(int width, int height, int maxSide) {
		if (maxSide <= 0) {
			return 1.0;
		}
		int longest = Math.max(width, height);
		return longest <= maxSide ? 1.0 : maxSide / (double) longest;
	}

	private static float[][] scaleQuad(float[][] quad, double factor) {
		float[][] out = new float[4][2];
		for (int i = 0; i < 4; i++) {
			out[i][0] = (float) (quad[i][0] * factor);
			out[i][1] = (float) (quad[i][1] * factor);
		}
		return out;
	}

	private static double longEdge(float[][] quad) {
		return Math.max(distance(quad[0], quad[1]), distance(quad[0], quad[3]));
	}

	private static double distance(float[] a, float[] b) {
		double dx = a[0] - b[0];
		double dy = a[1] - b[1];
		return Math.sqrt(dx * dx + dy * dy);
	}

	private static double distance(Point a, Point b) {
		double dx = a.x - b.x;
		double dy = a.y - b.y;
		return Math.sqrt(dx * dx + dy * dy);
	}

	private static Point[] toPoints(float[][] quad) {
		if (quad == null || quad.length != 4) {
			throw new MicaAiException(
				ErrorCode.CARD_FAILED,
				"四边形需为 4 个点，实际: " + (quad == null ? "null" : quad.length));
		}
		Point[] points = new Point[4];
		for (int i = 0; i < 4; i++) {
			points[i] = new Point(quad[i][0], quad[i][1]);
		}
		return points;
	}

	private static float[][] toQuad(Point[] points) {
		float[][] quad = new float[4][2];
		for (int i = 0; i < 4; i++) {
			quad[i][0] = (float) points[i].x;
			quad[i][1] = (float) points[i].y;
		}
		return quad;
	}

	private static double[] flatten(float[][] quad) {
		double[] data = new double[8];
		for (int i = 0; i < 4; i++) {
			data[i * 2] = quad[i][0];
			data[i * 2 + 1] = quad[i][1];
		}
		return data;
	}

	private static int rotateCode(int degrees) {
		switch (((degrees % 360) + 360) % 360) {
			case 90:
				return Core.ROTATE_90_CLOCKWISE;
			case 180:
				return Core.ROTATE_180;
			case 270:
				return Core.ROTATE_90_COUNTERCLOCKWISE;
			default:
				return -1;
		}
	}

	@Getter
	@Accessors(fluent = true)
	@AllArgsConstructor(access = AccessLevel.PACKAGE)
	static final class Quad {
		float[][] points;
		double area;
		double ratio;
		double ratioScore;
		double score;
	}

	@Getter
	@Accessors(fluent = true)
	@AllArgsConstructor(access = AccessLevel.PACKAGE)
	static final class Orientation {
		int degrees;
		boolean auto;
	}
}