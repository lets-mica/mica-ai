package net.dreamlu.mica.ai.voice.engine;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import lombok.extern.slf4j.Slf4j;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

/**
 * SenseVoice ONNX 编码器。
 *
 * <p>对应 Python 端 {@code SenseVoiceEncoder}：
 * <ul>
 *   <li>从 ONNX 模型元数据加载 lid_dict / textnorm_dict 等配置</li>
 *   <li>构造 4 个 Prompt Token ID</li>
 *   <li>执行 Encoder 推理，输出 (1, T+4, 512)</li>
 * </ul>
 */
@Slf4j
public final class SenseVoiceEncoder implements AutoCloseable {

	private final OrtEnvironment env;
	private final OrtSession session;

	/** 从模型元数据解析的配置 */
	private final Map<String, Integer> lidDict;
	private final Map<String, Integer> textnormDict;

	/** 模型输入精度 (float32 / float16 → 统一用 float32) */
	private final boolean useFp16;

	public SenseVoiceEncoder(String encoderPath, OrtEnvironment env) throws OrtException {
		this.env = env;

		OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
		opts.setIntraOpNumThreads(1);
		opts.setInterOpNumThreads(1);
		opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

		this.session = env.createSession(encoderPath, opts);

		// 从元数据加载配置
		Map<String, String> meta = session.getMetadata().getCustomMetadata();
		this.lidDict = parseSimpleDict(meta.getOrDefault("lid_dict", "{}"));
		this.textnormDict = parseSimpleDict(meta.getOrDefault("textnorm_dict", "{}"));

		// 检测精度
		String inputType = session.getInputInfo().values().iterator().next().toString();
		this.useFp16 = inputType.contains("float16");
		log.info("[Encoder] 初始化完成, fp16={}, lid_dict={}, textnorm_dict={}",
			useFp16, lidDict.keySet(), textnormDict);
	}

	/**
	 * 执行 Encoder 推理。
	 *
	 * @param lfrFeat LFR 特征 [T, 560]
	 * @param lid     语言 ID (如 "auto", "zh")
	 * @param itn     是否启用 ITN
	 * @return encoder 输出 float[1][T+4][512]
	 */
	public float[][][] forward(float[][] lfrFeat, String lid, boolean itn) throws OrtException {
		int tValid = lfrFeat.length;
		int inputSize = lfrFeat[0].length; // 560

		// 构造 prompt IDs
		long[] promptIds = constructPrompt(lid, itn);

		// 构造输入张量
		long[] featShape = {1, tValid, inputSize};
		FloatBuffer featBuf = FloatBuffer.allocate(tValid * inputSize);
		for (float[] row : lfrFeat) {
			featBuf.put(row);
		}
		featBuf.flip();

		long[] maskShape = {1, tValid};
		FloatBuffer maskBuf = FloatBuffer.allocate(tValid);
		for (int i = 0; i < tValid; i++) {
			maskBuf.put(1.0f);
		}
		maskBuf.flip();

		long[] promptShape = {1, 4};
		LongBuffer promptBuf = LongBuffer.wrap(promptIds);

		Map<String, OnnxTensor> inputs = new HashMap<>();
		inputs.put("speech_feat", OnnxTensor.createTensor(env, featBuf, featShape));
		inputs.put("mask", OnnxTensor.createTensor(env, maskBuf, maskShape));
		inputs.put("prompt_ids", OnnxTensor.createTensor(env, promptBuf, promptShape));

		try (OrtSession.Result result = session.run(inputs)) {
			OnnxTensor outTensor = (OnnxTensor) result.get(0);
			return readFloat3D(outTensor);
		}
	}

	/**
	 * 构造 4 个 Prompt Token ID: [lid_idx, 1, 2, itn_idx]
	 */
	private long[] constructPrompt(String lid, boolean itn) {
		// lid
		int lidIdx = lidDict.getOrDefault(lid, lidDict.getOrDefault("auto", 0));

		// itn
		String itnStr = itn ? "withitn" : "woitn";
		int itnIdx = textnormDict.getOrDefault(itnStr, 14);

		return new long[]{lidIdx, 1, 2, itnIdx};
	}

	@Override
	public void close() throws OrtException {
		if (session != null) {
			session.close();
		}
	}

	// ==================== JSON 简易解析 ====================

	/**
	 * 简易解析 JSON dict: {"key": intValue, ...}
	 */
	private static Map<String, Integer> parseSimpleDict(String json) {
		Map<String, Integer> result = new HashMap<>();
		if (json == null || json.isBlank()) return result;
		json = json.trim();
		if (json.startsWith("{")) {
			json = json.substring(1, json.length() - 1);
		}
		String[] pairs = json.split(",");
		for (String pair : pairs) {
			String[] kv = pair.split(":");
			if (kv.length == 2) {
				String k = kv[0].trim().replace("\"", "");
				try {
					int v = Integer.parseInt(kv[1].trim());
					result.put(k, v);
				} catch (NumberFormatException ignored) {
				}
			}
		}
		return result;
	}

	private static float[][][] readFloat3D(OnnxTensor tensor) throws OrtException {
		FloatBuffer buf = tensor.getFloatBuffer();
		long[] shape = tensor.getInfo().getShape();
		int d0 = (int) shape[0];
		int d1 = (int) shape[1];
		int d2 = (int) shape[2];
		float[] data = new float[d0 * d1 * d2];
		buf.get(data);
		float[][][] out = new float[d0][d1][d2];
		for (int i = 0; i < d0; i++) {
			for (int j = 0; j < d1; j++) {
				System.arraycopy(data, (i * d1 + j) * d2, out[i][j], 0, d2);
			}
		}
		return out;
	}
}

