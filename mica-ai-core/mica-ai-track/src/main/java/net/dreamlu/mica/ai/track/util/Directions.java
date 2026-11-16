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
package net.dreamlu.mica.ai.track.util;

/**
 * 屏幕坐标系下的八方向换算（y 轴向下）。
 */
public final class Directions {

	private static final String[] LABELS = {"右", "右下", "下", "左下", "左", "左上", "上", "右上"};

	private Directions() {
	}

	/**
	 * 速度向量对应角度：0° 向右、90° 向下，范围 [0, 360)。
	 */
	public static double degrees(double vx, double vy) {
		double deg = Math.toDegrees(Math.atan2(vy, vx));
		return deg < 0d ? deg + 360d : deg;
	}

	public static String label(double vx, double vy) {
		if (vx == 0d && vy == 0d) {
			return "静止";
		}
		int idx = (int) ((degrees(vx, vy) + 22.5d) / 45d) % LABELS.length;
		return LABELS[idx];
	}
}
