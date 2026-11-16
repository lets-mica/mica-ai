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
package net.dreamlu.mica.ai.track.event;

/**
 * 感兴趣区域（任意简单多边形，射线法判定内外）。
 *
 * <p>坐标为**原图像素**，与 {@link net.dreamlu.mica.ai.track.model.BBox} 同一坐标系。
 * 不可变。
 */
public class Roi {

	private final float[] xs;
	private final float[] ys;

	private Roi(float[] xs, float[] ys) {
		this.xs = xs;
		this.ys = ys;
	}

	/**
	 * 矩形 ROI，自动规整左上 / 右下角顺序。
	 */
	public static Roi rectangle(float x1, float y1, float x2, float y2) {
		float minX = Math.min(x1, x2);
		float maxX = Math.max(x1, x2);
		float minY = Math.min(y1, y2);
		float maxY = Math.max(y1, y2);
		return polygon(minX, minY, maxX, minY, maxX, maxY, minX, maxY);
	}

	/**
	 * 多边形 ROI，参数按 {@code x0,y0,x1,y1,...} 顺序给出，至少 3 个顶点。
	 */
	public static Roi polygon(float... xy) {
		if (xy == null || xy.length < 6 || xy.length % 2 != 0) {
			throw new IllegalArgumentException("多边形至少需要 3 个顶点，参数须为 x,y 成对");
		}
		int n = xy.length / 2;
		float[] xs = new float[n];
		float[] ys = new float[n];
		for (int i = 0; i < n; i++) {
			xs[i] = xy[i * 2];
			ys[i] = xy[i * 2 + 1];
		}
		return new Roi(xs, ys);
	}

	public boolean contains(float px, float py) {
		boolean inside = false;
		for (int i = 0, j = xs.length - 1; i < xs.length; j = i++) {
			float xi = xs[i];
			float yi = ys[i];
			float xj = xs[j];
			float yj = ys[j];
			boolean crosses = ((yi > py) != (yj > py))
				&& (px < (xj - xi) * (py - yi) / (yj - yi) + xi);
			if (crosses) {
				inside = !inside;
			}
		}
		return inside;
	}

	public int vertexCount() {
		return xs.length;
	}
}
