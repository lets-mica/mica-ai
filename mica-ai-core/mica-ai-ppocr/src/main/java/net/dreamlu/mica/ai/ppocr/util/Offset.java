package net.dreamlu.mica.ai.ppocr.util;

import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;
import org.locationtech.jts.geom.Coordinate;
import org.locationtech.jts.geom.Geometry;
import org.locationtech.jts.geom.GeometryFactory;
import org.locationtech.jts.geom.Polygon;
import org.locationtech.jts.operation.buffer.BufferOp;
import org.locationtech.jts.operation.buffer.BufferParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * 多边形偏移（unclip）。
 *
 * <p>对应 Python 端的 pyclipper {@code PyclipperOffset}：
 * <pre>
 *     po = pyclipper.PyclipperOffset()
 *     po.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
 *     expanded = np.asarray(po.Execute(distance))
 * </pre>
 *
 * <p>Java 实现基于 JTS（Java Topology Suite）的 {@link BufferOp}，使用
 * {@link BufferParameters#JOIN_ROUND} 圆角连接，与 pyclipper 行为接近。
 *
 * <p>注意：pyclipper 使用整数内部表示（带 scale 因子），而 JTS 使用 double 浮点。
 * 对典型 100~4000 像素范围的文本框，结果像素差异在 1px 以内。
 */
@Slf4j
@UtilityClass
public class Offset {
	private static final GeometryFactory GF = new GeometryFactory();

	/**
	 * 按指定距离对多边形进行外扩（unclip）。
	 *
	 * @param polygon  (N, 2) 顶点列表（已按 [TL, TR, BR, BL] 等任意顺序）
	 * @param distance 扩展距离（像素），正数外扩，负数内缩
	 * @return 扩展后多边形顶点 (M, 2)；若扩展失败或结果为空，回退返回原多边形
	 */
	public static float[][] unclip(float[][] polygon, double distance) {
		if (polygon == null || polygon.length < 3) {
			return polygon;
		}

		Coordinate[] coords = new Coordinate[polygon.length + 1];
		for (int i = 0; i < polygon.length; i++) {
			coords[i] = new Coordinate(polygon[i][0], polygon[i][1]);
		}
		coords[polygon.length] = coords[0]; // 闭合

		Polygon poly;
		try {
			poly = GF.createPolygon(coords);
		} catch (Exception e) {
			log.debug("Offset: 无法构造多边形, 跳过: {}", e.getMessage());
			return polygon;
		}

		BufferParameters params = new BufferParameters();
		params.setJoinStyle(BufferParameters.JOIN_ROUND);
		// End cap 对闭合多边形无效，无需设置

		Geometry result;
		try {
			result = BufferOp.bufferOp(poly, distance, params);
		} catch (Exception e) {
			log.debug("Offset: BufferOp 失败, 尝试简化输入: {}", e.getMessage());
			result = BufferOp.bufferOp(poly, distance);
		}

		if (result == null || result.isEmpty()) {
			log.debug("Offset: 结果为空, 保留原多边形");
			return polygon;
		}

		// 选择面积最大的多边形
		Polygon best = null;
		double bestArea = -1.0;
		if (result instanceof Polygon p) {
			best = p;
			bestArea = p.getArea();
		} else if (result instanceof org.locationtech.jts.geom.MultiPolygon mp) {
			for (int i = 0; i < mp.getNumGeometries(); i++) {
				Polygon g = (Polygon) mp.getGeometryN(i);
				double a = g.getArea();
				if (a > bestArea) {
					bestArea = a;
					best = g;
				}
			}
		} else {
			// 兜底：取凸包
			Geometry convex = result.convexHull();
			if (convex instanceof Polygon cp) {
				best = cp;
				bestArea = cp.getArea();
			} else {
				return polygon;
			}
		}

		if (best == null) {
			return polygon;
		}

		Coordinate[] exterior = best.getExteriorRing().getCoordinates();
		// 去除尾部的重复闭合点
		int n = exterior.length;
		if (n > 1 && exterior[0].equals(exterior[n - 1])) {
			n--;
		}
		float[][] out = new float[n][2];
		for (int i = 0; i < n; i++) {
			out[i][0] = (float) exterior[i].x;
			out[i][1] = (float) exterior[i].y;
		}
		return out;
	}

	/**
	 * 计算多边形面积（使用 OpenCV {@code contourArea}，与 pyclipper 的输入计算保持一致）。
	 *
	 * @param polygon (N, 2) 顶点
	 * @return 面积（像素²）
	 */
	public static double area(float[][] polygon) {
		if (polygon == null || polygon.length < 3) {
			return 0.0;
		}
		org.opencv.core.Point[] pts = new org.opencv.core.Point[polygon.length];
		for (int i = 0; i < polygon.length; i++) {
			pts[i] = new org.opencv.core.Point(polygon[i][0], polygon[i][1]);
		}
		org.opencv.core.MatOfPoint2f mop = new org.opencv.core.MatOfPoint2f();
		mop.fromArray(pts);
		return org.opencv.imgproc.Imgproc.contourArea(mop);
	}

	/**
	 * 计算闭合多边形周长（使用 OpenCV {@code arcLength}）。
	 *
	 * @param polygon (N, 2) 顶点
	 * @return 周长（像素）
	 */
	public static double perimeter(float[][] polygon) {
		if (polygon == null || polygon.length < 2) {
			return 0.0;
		}
		org.opencv.core.Point[] pts = new org.opencv.core.Point[polygon.length];
		for (int i = 0; i < polygon.length; i++) {
			pts[i] = new org.opencv.core.Point(polygon[i][0], polygon[i][1]);
		}
		org.opencv.core.MatOfPoint2f mop = new org.opencv.core.MatOfPoint2f();
		mop.fromArray(pts);
		return org.opencv.imgproc.Imgproc.arcLength(mop, true);
	}

	/**
	 * 计算 unclip 距离：{@code area * unclipRatio / length}，与 pyclipper 等价。
	 */
	public static double unclipDistance(float[][] polygon, double unclipRatio) {
		double a = area(polygon);
		double l = perimeter(polygon);
		if (l < 1e-6) {
			return 0.0;
		}
		return a * unclipRatio / l;
	}

	/**
	 * 静默吞掉空几何，返回 true 表示结果有意义。
	 */
	private static boolean isValid(Geometry g) {
		return g != null && !g.isEmpty() && g.getArea() > 0.0;
	}

	/**
	 * 调试辅助：列出所有候选几何。
	 */
	public static List<float[][]> collectAll(Geometry result) {
		List<float[][]> all = new ArrayList<>();
		if (result instanceof Polygon p) {
			all.add(toFloat(p.getExteriorRing().getCoordinates()));
		} else if (result instanceof org.locationtech.jts.geom.MultiPolygon mp) {
			for (int i = 0; i < mp.getNumGeometries(); i++) {
				Polygon g = (Polygon) mp.getGeometryN(i);
				all.add(toFloat(g.getExteriorRing().getCoordinates()));
			}
		}
		return all;
	}

	private static float[][] toFloat(Coordinate[] coords) {
		int n = coords.length;
		if (n > 1 && coords[0].equals(coords[n - 1])) {
			n--;
		}
		float[][] out = new float[n][2];
		for (int i = 0; i < n; i++) {
			out[i][0] = (float) coords[i].x;
			out[i][1] = (float) coords[i].y;
		}
		return out;
	}
}
