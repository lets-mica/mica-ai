package net.dreamlu.mica.ai.tts.internal;

import ai.onnxruntime.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.*;

/**
 * Kokoro ONNX 推理引擎。
 * <p>封装 ONNX Runtime 会话管理，执行模型推理。
 */
public final class KokoroEngine implements AutoCloseable {

	private static final Logger log = LoggerFactory.getLogger(KokoroEngine.class);

	private final OrtEnvironment env;
	private final OrtSession session;

	public KokoroEngine(String modelPath, String provider) throws OrtException {
		this.env = OrtEnvironment.getEnvironment();
		OrtSession.SessionOptions options = new OrtSession.SessionOptions();

		if ("cuda".equalsIgnoreCase(provider) || "gpu".equalsIgnoreCase(provider)) {
			try {
				options.addCUDA(0);
				log.info("Using CUDA provider");
			} catch (OrtException e) {
				log.warn("CUDA not available, falling back to CPU: {}", e.getMessage());
			}
		} else {
			log.info("Using CPU provider");
		}

		this.session = env.createSession(modelPath, options);
		log.info("Loaded model: {}", modelPath);
	}

	/**
	 * 执行推理。
	 *
	 * @param inputIds token ID 序列 [1, seq_len]（含首尾 padding 0）
	 * @param refS     style 向量 [1, 256]
	 * @param speed    语速 [1]
	 * @return 音频数据 float[]
	 */
	public float[] inference(long[] inputIds, float[] refS, float speed) throws OrtException {
		// 构造 input_ids tensor [1, seq_len]
		int seqLen = inputIds.length;
		LongBuffer inputIdsBuf = LongBuffer.wrap(inputIds);
		OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIdsBuf, new long[]{1, seqLen});

		// 构造 ref_s tensor [1, 256]
		FloatBuffer refSBuf = FloatBuffer.wrap(refS);
		OnnxTensor refSTensor = OnnxTensor.createTensor(env, refSBuf, new long[]{1, refS.length});

		// 构造 speed tensor [1]
		FloatBuffer speedBuf = FloatBuffer.wrap(new float[]{speed});
		OnnxTensor speedTensor = OnnxTensor.createTensor(env, speedBuf, new long[]{1});

		Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
		inputs.put("input_ids", inputIdsTensor);
		inputs.put("ref_s", refSTensor);
		inputs.put("speed", speedTensor);

		try (OrtSession.Result result = session.run(inputs)) {
			// 获取 audio 输出
			OnnxTensor audioTensor = (OnnxTensor) result.get("audio").orElseThrow();
			float[] audio = audioTensor.getFloatBuffer().array();

			// 可选：获取 pred_dur
			OnnxTensor durTensor = (OnnxTensor) result.get("pred_dur").orElse(null);
			if (durTensor != null) {
				long[] dur = durTensor.getLongBuffer().array();
				log.debug("Predicted durations: {} tokens", dur.length);
			}

			return audio;
		} finally {
			inputIdsTensor.close();
			refSTensor.close();
			speedTensor.close();
		}
	}

	@Override
	public void close() throws OrtException {
		if (session != null) {
			session.close();
		}
	}
}
