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
package net.dreamlu.mica.ai.barcode.decoding;

import com.google.zxing.BarcodeFormat;
import com.google.zxing.BinaryBitmap;
import com.google.zxing.Binarizer;
import com.google.zxing.DecodeHintType;
import com.google.zxing.LuminanceSource;
import com.google.zxing.MultiFormatReader;
import com.google.zxing.NotFoundException;
import com.google.zxing.Result;
import com.google.zxing.ResultPoint;
import com.google.zxing.common.GlobalHistogramBinarizer;
import com.google.zxing.common.HybridBinarizer;
import lombok.RequiredArgsConstructor;
import net.dreamlu.mica.ai.barcode.config.BarcodeConfig;
import net.dreamlu.mica.ai.barcode.model.BarcodeImage;
import net.dreamlu.mica.ai.barcode.model.BarcodeResult;
import net.dreamlu.mica.ai.barcode.model.BarcodeSymbology;
import net.dreamlu.mica.ai.barcode.util.BarcodeImageUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * 解码链。策略顺序来自选型实测结论（已固化为单测 {@code BarcodeEngineTest}）：
 * {@code Hybrid + TRY_HARDER} → {@code ×2 / ×3 最近邻放大} → {@code 2×2 带重叠分块}
 * → {@code GlobalHistogramBinarizer} 回退。
 *
 * <p>「多尺度」与「分块」是性价比最高的两项增强（1px 小码 0→100%、模糊 σ=2 的 0→100%、
 * 一图 4 码 25→100%），且都不需要任何 CNN 定位模型。固定单一二值化器是错的：模糊 σ=2 时
 * GlobalHistogram 还有 33.3%、Hybrid 反而 0%，所以末档保留回退。
 *
 * <p>线程安全：{@link MultiFormatReader} 有状态，本类每次解码新建一个；构造后冻结的
 * {@code hints} 只被读取，可作为 Spring 单例共享。
 */
public class BarcodeDecoder {

	private final BarcodeConfig config;
	private final Map<DecodeHintType, Object> hints;

	public BarcodeDecoder(BarcodeConfig config) {
		if (config == null) {
			throw new IllegalArgumentException("BarcodeConfig must not be null");
		}
		config.validate();
		this.config = config;
		this.hints = buildHints(config);
	}

	public List<BarcodeResult> decode(BarcodeImage image) {
		return decodeInternal(image, false);
	}

	public Optional<BarcodeResult> decodeFirst(BarcodeImage image) {
		List<BarcodeResult> results = decodeInternal(image, true);
		return results.isEmpty() ? Optional.empty() : Optional.of(results.get(0));
	}

	private List<BarcodeResult> decodeInternal(BarcodeImage image, boolean firstOnly) {
		if (image == null) {
			return Collections.emptyList();
		}
		Map<String, BarcodeResult> found = new LinkedHashMap<>();
		for (View view : views(image)) {
			Result raw = readOne(view);
			if (raw == null) {
				continue;
			}
			BarcodeSymbology symbology = BarcodeSymbology.ofName(raw.getBarcodeFormat().name());
			if (symbology == null || raw.getText() == null) {
				continue;
			}
			String key = symbology.name() + '\u0000' + raw.getText();
			if (found.containsKey(key)) {
				continue;
			}
			found.put(key, toResult(raw, symbology, view, image));
			if (firstOnly || found.size() >= config.getMaxResults()) {
				break;
			}
		}
		return new ArrayList<>(found.values());
	}

	private Result readOne(View view) {
		LuminanceSource source = BarcodeImageUtils.toLuminanceSource(view.image);
		Binarizer binarizer = view.globalHistogram
			? new GlobalHistogramBinarizer(source) : new HybridBinarizer(source);
		try {
			return new MultiFormatReader().decode(new BinaryBitmap(binarizer), hints);
		} catch (NotFoundException e) {
			return null;
		}
	}

	private List<View> views(BarcodeImage original) {
		List<View> out = new ArrayList<>();
		out.add(new View(original, 1d, 0, 0, "DIRECT", false));
		if (config.isMultiScale()) {
			for (Integer factor : config.getUpscaleFactors()) {
				if (factor == null || !fitsMaxSide(original, factor)) {
					continue;
				}
				out.add(new View(BarcodeImageUtils.upscaleNearest(original, factor),
					factor, 0, 0, "UPSCALE_x" + factor, false));
			}
		}
		if (config.isTileScan() && (config.getTileRows() > 1 || config.getTileCols() > 1)) {
			int width = original.getWidth();
			int height = original.getHeight();
			int tileWidth = width / config.getTileCols();
			int tileHeight = height / config.getTileRows();
			int overlap = (int) (Math.min(tileWidth, tileHeight) * config.getTileOverlapRatio());
			if (tileWidth > 0 && tileHeight > 0) {
				for (int row = 0; row < config.getTileRows(); row++) {
					for (int col = 0; col < config.getTileCols(); col++) {
						int x1 = Math.max(0, col * tileWidth - overlap);
						int y1 = Math.max(0, row * tileHeight - overlap);
						int x2 = Math.min(width, (col + 1) * tileWidth + overlap);
						int y2 = Math.min(height, (row + 1) * tileHeight + overlap);
						BarcodeImage tile = BarcodeImageUtils.crop(original, x1, y1, x2, y2);
						String name = "TILE_" + row + "_" + col;
						out.add(new View(tile, 1d, x1, y1, name, false));
						if (config.isMultiScale()) {
							for (Integer factor : config.getUpscaleFactors()) {
								if (factor == null || !fitsMaxSide(tile, factor)) {
									continue;
								}
								out.add(new View(BarcodeImageUtils.upscaleNearest(tile, factor),
									factor, x1, y1, name + "_UPSCALE_x" + factor, false));
							}
						}
					}
				}
			}
		}
		if (config.isGlobalHistogramFallback()) {
			out.add(new View(original, 1d, 0, 0, "GLOBAL_HISTOGRAM", true));
		}
		return out;
	}

	private boolean fitsMaxSide(BarcodeImage image, int factor) {
		long scaled = (long) Math.max(image.getWidth(), image.getHeight()) * factor;
		return scaled <= config.getMaxSideLength();
	}

	private BarcodeResult toResult(Result raw, BarcodeSymbology symbology,
								   View view, BarcodeImage original) {
		ResultPoint[] points = raw.getResultPoints();
		int[] box = null;
		if (points != null && points.length > 0) {
			double minX = Double.MAX_VALUE;
			double minY = Double.MAX_VALUE;
			double maxX = -Double.MAX_VALUE;
			double maxY = -Double.MAX_VALUE;
			for (ResultPoint point : points) {
				double x = point.getX() / view.scale + view.offsetX;
				double y = point.getY() / view.scale + view.offsetY;
				minX = Math.min(minX, x);
				minY = Math.min(minY, y);
				maxX = Math.max(maxX, x);
				maxY = Math.max(maxY, y);
			}
			box = new int[]{
				clamp(minX, original.getWidth()),
				clamp(minY, original.getHeight()),
				clamp(maxX, original.getWidth()),
				clamp(maxY, original.getHeight())
			};
		}
		return new BarcodeResult(raw.getText(), symbology, config.getCharset(), box, view.strategy);
	}

	private static int clamp(double value, int upper) {
		if (Double.isNaN(value) || Double.isInfinite(value)) {
			return 0;
		}
		int rounded = (int) Math.round(value);
		return rounded < 0 ? 0 : (Math.min(rounded, upper));
	}

	private static Map<DecodeHintType, Object> buildHints(BarcodeConfig config) {
		EnumSet<BarcodeFormat> formats = EnumSet.noneOf(BarcodeFormat.class);
		for (BarcodeSymbology symbology : config.getFormats()) {
			formats.add(BarcodeFormat.valueOf(symbology.name()));
		}
		Map<DecodeHintType, Object> hints = new EnumMap<>(DecodeHintType.class);
		hints.put(DecodeHintType.POSSIBLE_FORMATS, formats);
		hints.put(DecodeHintType.CHARACTER_SET, config.getCharset());
		if (config.isTryHarder()) {
			hints.put(DecodeHintType.TRY_HARDER, Boolean.TRUE);
		}
		return Collections.unmodifiableMap(hints);
	}

	@RequiredArgsConstructor
	private static final class View {
		private final BarcodeImage image;
		private final double scale;
		private final int offsetX;
		private final int offsetY;
		private final String strategy;
		private final boolean globalHistogram;
	}
}
