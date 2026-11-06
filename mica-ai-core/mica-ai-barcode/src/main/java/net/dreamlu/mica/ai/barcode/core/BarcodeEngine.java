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
package net.dreamlu.mica.ai.barcode.core;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.barcode.config.BarcodeConfig;
import net.dreamlu.mica.ai.barcode.decoding.BarcodeDecoder;
import net.dreamlu.mica.ai.barcode.model.BarcodeImage;
import net.dreamlu.mica.ai.barcode.model.BarcodeResult;
import net.dreamlu.mica.ai.barcode.model.BarcodeSymbology;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;

/**
 * 二维码 / 一维条码解码门面。零模型、零 ONNX、零 Spring、零 OpenCV，开箱即用：
 *
 * <pre>{@code
 * BarcodeEngine engine = BarcodeEngine.defaults();
 * List<BarcodeResult> codes = engine.decodeBytes(file.getBytes());   // 一图多码
 * Optional<BarcodeResult> one = engine.decodeFirstBytes(bytes);      // 只要一个码，命中即短路
 * }</pre>
 *
 * <p>本引擎<b>不实现 {@code AutoCloseable}</b>：不持有 ONNX session、{@code Mat} 或文件句柄，
 * 没有需要释放的资源，Spring Bean 也无需 {@code destroyMethod}。构造后配置冻结，线程安全，
 * 可作为单例共享；解码内部每次新建 {@code MultiFormatReader}。
 *
 * <p>异常语义：与 {@code mica-ai-track} 一致刻意零依赖（不引 {@code mica-ai-common}），
 * 参数 / 文件 / 图像解码错误统一抛 JDK 原生 {@link IllegalArgumentException}。
 */
@Slf4j
public class BarcodeEngine {

	@Getter
	private final BarcodeConfig config;
	private final BarcodeDecoder decoder;

	public BarcodeEngine(BarcodeConfig config) {
		if (config == null) {
			throw new IllegalArgumentException("BarcodeConfig must not be null");
		}
		config.validate();
		this.config = config;
		this.decoder = new BarcodeDecoder(config);
		log.info("mica-ai-barcode 初始化完成: formats={} multiScale={} tileScan={} globalFallback={}",
			config.getFormats().size(), config.isMultiScale(),
			config.isTileScan(), config.isGlobalHistogramFallback());
	}

	public static BarcodeEngine create(BarcodeConfig config) {
		return new BarcodeEngine(config);
	}

	/**
	 * 零配置开箱即用（本能力没有模型路径这类必填项）。
	 */
	public static BarcodeEngine defaults() {
		return new BarcodeEngine(BarcodeConfig.DEFAULTS);
	}

	public List<BarcodeResult> decode(BarcodeImage image) {
		return decoder.decode(image);
	}

	public Optional<BarcodeResult> decodeFirst(BarcodeImage image) {
		return decoder.decodeFirst(image);
	}

	public List<BarcodeResult> decodeBytes(byte[] imageBytes) {
		return decoder.decode(BarcodeImage.fromEncodedBytes(imageBytes));
	}

	public Optional<BarcodeResult> decodeFirstBytes(byte[] imageBytes) {
		return decoder.decodeFirst(BarcodeImage.fromEncodedBytes(imageBytes));
	}

	public List<BarcodeResult> decodePath(String imagePath) {
		return decodeBytes(readFile(imagePath));
	}

	public Optional<BarcodeResult> decodeFirstPath(String imagePath) {
		return decodeFirstBytes(readFile(imagePath));
	}

	public List<BarcodeSymbology> formats() {
		return config.getFormats();
	}

	private static byte[] readFile(String imagePath) {
		if (imagePath == null || imagePath.trim().isEmpty()) {
			throw new IllegalArgumentException("imagePath 不能为空");
		}
		Path path = Paths.get(imagePath);
		if (!Files.exists(path) || !Files.isRegularFile(path)) {
			throw new IllegalArgumentException("图像文件不存在: " + imagePath);
		}
		try {
			return Files.readAllBytes(path);
		} catch (IOException e) {
			throw new IllegalArgumentException("读取图像失败: " + imagePath, e);
		}
	}
}
