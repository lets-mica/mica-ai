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
package net.dreamlu.mica.ai.filetype.detection;

import ai.onnxruntime.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.onnx.OrtDevice;
import net.dreamlu.mica.ai.common.onnx.OrtSessionOptions;
import net.dreamlu.mica.ai.common.util.IOUtil;
import net.dreamlu.mica.ai.filetype.config.ContentTypeRegistry;
import net.dreamlu.mica.ai.filetype.config.FiletypeConfig;
import net.dreamlu.mica.ai.filetype.config.ModelConfig;
import net.dreamlu.mica.ai.filetype.config.PredictionMode;
import net.dreamlu.mica.ai.filetype.feature.FeaturesExtractor;
import net.dreamlu.mica.ai.filetype.model.ContentTypeInfo;
import net.dreamlu.mica.ai.filetype.model.ContentTypeLabel;
import net.dreamlu.mica.ai.filetype.model.FiletypeResult;
import net.dreamlu.mica.ai.filetype.postprocess.PredictionPostProcessor;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.channels.SeekableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * 文件类型识别引擎（Google Magika {@code standard_v3_3} 的 Java 复刻）。
 *
 * <p>零 Spring、零 Python，纯 ONNX Runtime 推理。用法：
 * <pre>{@code
 * try (FiletypeDetector detector = FiletypeDetector.create(config)) {
 *     FiletypeResult result = detector.detectPath("Dockerfile");
 *     System.out.println(result.getOutputLabel() + " " + result.getScore());
 * }
 * }</pre>
 *
 * <p>处理流程与 Python 版一致：空文件 / 超小文件走规则分支，其余提取
 * beg+end 特征送入模型，argmax 后按预测模式做后处理。
 */
@Slf4j
public class FiletypeDetector implements AutoCloseable {

	private static final String DEFAULT_MODEL_VERSION = "standard_v3_3";

	private final OrtEnvironment environment;
	private final OnnxModelSession modelSession;
	private final OrtSession.SessionOptions sessionOptions;
	@Getter
	private final ModelConfig modelConfig;
	@Getter
	private final ContentTypeRegistry contentTypeRegistry;
	private final PredictionPostProcessor postProcessor;
	private final PredictionMode predictionMode;
	private final String inputName;

	public FiletypeDetector(FiletypeConfig config) {
		Objects.requireNonNull(config, "FiletypeConfig must not be null");
		config.validate();
		String version = resolveModelVersion(config.getModelVersion());
		this.predictionMode = config.getPredictionMode();
		this.environment = OrtEnvironment.getEnvironment();
		this.sessionOptions = buildSessionOptions(config.getOnnx());
		this.modelSession = new OnnxModelSession(
			environment, config.resolveModelPath(),
			sessionOptions, "filetype");
		this.modelConfig = loadModelConfig(config.resolveConfigPath());
		this.contentTypeRegistry = ContentTypeRegistry.load(
			config.resolveContentTypesPath());
		this.postProcessor = new PredictionPostProcessor(modelConfig, contentTypeRegistry, predictionMode);
		this.inputName = modelSession.getSession().getInputNames().iterator().next();
		log.info("mica-ai-filetype 初始化完成: version={} labels={} features={}",
			version, modelConfig.getTargetLabelsSpace().size(), modelConfig.featuresSize());
	}

	public static FiletypeDetector create(FiletypeConfig config) {
		return new FiletypeDetector(config);
	}

	public static FiletypeDetector createDefault() {
		return new FiletypeDetector(FiletypeConfig.builder().build());
	}

	public FiletypeResult detectPath(String path) {
		if (path == null || path.isEmpty()) {
			throw new MicaAiException(
				ErrorCode.ILLEGAL_ARGUMENT, "待检测路径为空");
		}
		return detectPath(Paths.get(path));
	}

	public FiletypeResult detectPath(Path path) {
		Objects.requireNonNull(path, "Path must not be null");
		try {
			if (Files.isDirectory(path)) {
				return specialResult(ContentTypeLabel.DIRECTORY);
			}
			if (!Files.isRegularFile(path)) {
				if (!Files.exists(path)) {
					throw new MicaAiException(
						ErrorCode.NOT_FOUND, "文件不存在: " + path);
				}
				return specialResult(ContentTypeLabel.UNKNOWN);
			}
			if (!Files.isReadable(path)) {
				throw new MicaAiException(
					ErrorCode.ILLEGAL_ARGUMENT, "文件不可读: " + path);
			}
			try (SeekableByteChannel channel = Files.newByteChannel(path)) {
				return detect(new ChannelSeekable(channel, Files.size(path)));
			}
		} catch (IOException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "读取文件失败: " + path, e);
		}
	}

	public FiletypeResult detectBytes(byte[] content) {
		if (content == null) {
			throw new MicaAiException(
				ErrorCode.ILLEGAL_ARGUMENT, "待检测字节数组为空");
		}
		return detect(new ByteArraySeekable(content));
	}

	/**
	 * 从输入流识别。流长度未知，会先完整读入内存，超大内容请改用
	 * {@link #detectPath(String)} 或 {@link #detectBytes(byte[])}。
	 */
	public FiletypeResult detectStream(InputStream stream) {
		Objects.requireNonNull(stream, "InputStream must not be null");
		try {
			return detectBytes(IOUtil.readAllBytes(stream));
		} catch (IOException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "读取输入流失败", e);
		}
	}

	private FiletypeResult detect(Seekable src) {
		try {
			long size = src.size();
			if (size == 0) {
				return specialResult(ContentTypeLabel.EMPTY);
			}
			int block = modelConfig.getBlockSize();
			if (size < modelConfig.getMinFileSizeForDl()) {
				return fewBytesResult(src.readAt(0, (int) Math.min(size, (long) block)));
			}
			int n = (int) Math.min(size, (long) block);
			byte[] head = src.readAt(0, n);
			byte[] tail = size <= block ? head : src.readAt(size - n, n);
			int[] features = FeaturesExtractor.extract(modelConfig, head, tail);
			if (!FeaturesExtractor.hasEnoughMeaningfulBytes(modelConfig, features)) {
				return fewBytesResult(head);
			}
			float[] scores = runInference(features);
			int argmax = argmax(scores);
			String dlLabel = labelAt(argmax);
			float score = scores[argmax];
			String outputLabel = postProcessor.resolveOutputLabel(dlLabel, score);
			return buildResult(dlLabel, outputLabel, score);
		} catch (IOException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "读取待检测内容失败", e);
		}
	}

	private FiletypeResult fewBytesResult(byte[] content) {
		return specialResult(labelFromFewBytes(content));
	}

	static String labelFromFewBytes(byte[] content) {
		try {
			StandardCharsets.UTF_8.newDecoder()
				.decode(java.nio.ByteBuffer.wrap(content));
			return ContentTypeLabel.TXT;
		} catch (Exception e) {
			return ContentTypeLabel.UNKNOWN;
		}
	}

	private float[] runInference(int[] features) {
		// ONNX Runtime OrtSession thread-safe；OnnxTensor 每次新建不在并发路径共享，无需额外锁。
		try (OnnxTensor tensor = OnnxTensor.createTensor(
			environment, java.nio.IntBuffer.wrap(features), new long[]{1, features.length})) {
			Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
			try (OrtSession.Result result = modelSession.getSession().run(inputs)) {
				return toFloats(result.get(0).getValue());
			}
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.INFERENCE_FAILED, "filetype 模型推理失败", e);
		}
	}

	private static float[] toFloats(Object value) {
		if (value instanceof float[][]) {
			return ((float[][]) value)[0].clone();
		}
		if (value instanceof float[]) {
			return ((float[]) value).clone();
		}
		throw new MicaAiException(
			ErrorCode.INFERENCE_FAILED,
			"不支持的模型输出类型: " + (value == null ? "null" : value.getClass().getName()));
	}

	private String labelAt(int index) {
		List<String> labels = modelConfig.getTargetLabelsSpace();
		if (labels == null || index < 0 || index >= labels.size()) {
			return ContentTypeLabel.UNKNOWN;
		}
		return labels.get(index);
	}

	static int argmax(float[] scores) {
		int best = 0;
		for (int i = 1; i < scores.length; i++) {
			if (scores[i] > scores[best]) {
				best = i;
			}
		}
		return best;
	}

	private FiletypeResult buildResult(String dlLabel, String outputLabel, float score) {
		ContentTypeInfo info = contentTypeRegistry.get(outputLabel);
		return new FiletypeResult(outputLabel, dlLabel, score, info, predictionMode, info.isText());
	}

	private FiletypeResult specialResult(String label) {
		ContentTypeInfo info = contentTypeRegistry.get(label);
		return new FiletypeResult(label, ContentTypeLabel.UNDEFINED, 1.0f, info, predictionMode, info.isText());
	}

	private static String resolveModelVersion(String version) {
		return version == null || version.isEmpty() ? DEFAULT_MODEL_VERSION : version;
	}

	private static ModelConfig loadModelConfig(String path) {
		ObjectMapper mapper = new ObjectMapper();
		try {
			if (OnnxModelSession.isClasspath(path)) {
				String resource = OnnxModelSession.stripClasspathPrefix(path);
				return mapper.readValue(OnnxModelSession.loadClasspathBytes(resource), ModelConfig.class);
			}
			try (InputStream in = Files.newInputStream(Paths.get(path))) {
				return mapper.readValue(in, ModelConfig.class);
			}
		} catch (IOException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "加载模型配置失败: " + path, e);
		}
	}

	private static OrtSession.SessionOptions buildSessionOptions(OrtSessionOptions options) {
		OrtSessionOptions opts = options != null ? options : OrtSessionOptions.defaults();
		OrtSession.SessionOptions so = new OrtSession.SessionOptions();
		try {
			if (opts.getIntraOpNumThreads() > 0) {
				so.setIntraOpNumThreads(opts.getIntraOpNumThreads());
			}
			if (opts.getInterOpNumThreads() > 0) {
				so.setInterOpNumThreads(opts.getInterOpNumThreads());
			}
			so.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
		} catch (OrtException e) {
			throw new MicaAiException(
				ErrorCode.MODEL_LOAD_FAILED, "配置 ONNX 会话选项失败", e);
		}
		OrtDevice device = opts.getDevice();
		if (OrtDevice.GPU == device) {
			try {
				if (OrtEnvironment.getAvailableProviders().contains(OrtProvider.CUDA)) {
					so.addCUDA(opts.getCudaDeviceId());
				} else {
					log.warn("mica-ai-filetype: 未检测到可用的 CUDA 执行提供器，回退到 CPU 推理");
				}
			} catch (OrtException e) {
				log.warn("mica-ai-filetype: 启用 CUDA 执行提供器失败，回退到 CPU 推理: {}", e.getMessage());
			}
		}
		return so;
	}

	@Override
	public void close() {
		modelSession.close();
		try {
			sessionOptions.close();
		} catch (Exception e) {
			log.warn("关闭共享 OrtSession.SessionOptions 失败: {}", e.getMessage());
		}
	}

	private interface Seekable {

		long size() throws IOException;

		byte[] readAt(long offset, int length) throws IOException;
	}

	private static final class ByteArraySeekable implements Seekable {

		private final byte[] content;

		ByteArraySeekable(byte[] content) {
			this.content = content;
		}

		@Override
		public long size() {
			return content.length;
		}

		@Override
		public byte[] readAt(long offset, int length) {
			int from = (int) offset;
			int to = Math.min(content.length, from + length);
			return Arrays.copyOfRange(content, from, to);
		}
	}

	private static final class ChannelSeekable implements Seekable {

		private final SeekableByteChannel channel;
		private final long size;

		ChannelSeekable(SeekableByteChannel channel, long size) {
			this.channel = channel;
			this.size = size;
		}

		@Override
		public long size() {
			return size;
		}

		@Override
		public byte[] readAt(long offset, int length) throws IOException {
			channel.position(offset);
			ByteBuffer buffer = ByteBuffer.allocate(length);
			int off = 0;
			while (off < length) {
				int n = channel.read(buffer);
				if (n < 0) {
					break;
				}
				off += n;
			}
			if (off == length) {
				return buffer.array();
			}
			return Arrays.copyOf(buffer.array(), off);
		}
	}
}
