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

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 支持的条码制式。枚举常量名与 ZXing {@code BarcodeFormat} 常量名严格一致，
 * 因此公开 API 不泄漏 ZXing 类型，映射在解码器内部按名字完成。
 *
 * <p>{@link #DEFAULT_FORMATS} 是默认启用的保守集合：覆盖票据 / 面单 / 商品标签的高频制式，
 * 不开 Data Matrix / PDF417 / Aztec（未做退化样本验证，且全开会显著拖慢 TRY_HARDER）。
 */
@Getter
@RequiredArgsConstructor
public enum BarcodeSymbology {

	QR_CODE("QR 二维码", false),
	DATA_MATRIX("Data Matrix", false),
	PDF_417("PDF417", false),
	AZTEC("Aztec", false),
	CODE_128("Code 128", true),
	CODE_39("Code 39", true),
	CODE_93("Code 93", true),
	EAN_13("EAN-13", true),
	EAN_8("EAN-8", true),
	UPC_A("UPC-A", true),
	UPC_E("UPC-E", true),
	ITF("ITF 交叉二五码", true),
	CODABAR("Codabar", true);

	public static final List<BarcodeSymbology> DEFAULT_FORMATS = Collections.unmodifiableList(
		Arrays.asList(QR_CODE, CODE_128, EAN_13, EAN_8, UPC_A, CODE_39, ITF, CODABAR));

	private final String label;
	private final boolean oneDimensional;

	public static BarcodeSymbology ofName(String name) {
		if (name == null) {
			return null;
		}
		for (BarcodeSymbology value : values()) {
			if (value.name().equals(name)) {
				return value;
			}
		}
		return null;
	}
}
