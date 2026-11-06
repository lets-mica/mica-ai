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
package net.dreamlu.mica.ai.barcode.model;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.ToString;

/**
 * 单个条码的解码结果，不可变、不持有任何原生资源。
 *
 * <p>{@link #boundingBox} 由 ZXing 的定位点反推并映射回<b>原图坐标</b>（已还原放大倍数与分块偏移），
 * 但定位点通常只覆盖定位图形（QR 的三个角、一维码的起止点），因此它是<b>近似包围盒</b>，
 * 不保证完整框住整个码；需要精确框请自行做角点检测。
 */
@Getter
@ToString
@AllArgsConstructor
public class BarcodeResult {

	private final String text;
	private final BarcodeSymbology symbology;
	private final String charset;
	private final int[] boundingBox;
	private final String strategy;

	public boolean hasBoundingBox() {
		return boundingBox != null && boundingBox.length == 4;
	}
}
