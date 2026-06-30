package net.dreamlu.mica.ai.tts;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import net.dreamlu.mica.ai.tts.config.KokoroTtsConfig;
import net.dreamlu.mica.ai.tts.config.TtsResult;
import net.dreamlu.mica.ai.tts.engine.KokoroTts;
import org.junit.jupiter.api.Test;

import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 诊断：将 KokoroTts 实际送入 ONNX 的 input_ids / ref_s / speed 落盘，
 *       方便 Python 用同样输入直接推理对比。
 */
class KokoroTtsDumpInputsTest {

	private static final String MODEL_DIR = "E:\\codes\\ai\\kokoro-onnx\\model";
	private static final String OUT_DIR = "E:\\codes\\ai\\mica-ai\\model-tools";

	@Test
	void dumpInputsAndAudio() throws Exception {
		Path modelPath = Path.of(MODEL_DIR, "model_dynamic.onnx");
		Path voicesDir = Path.of(MODEL_DIR, "voices");
		Path configPath = Path.of(MODEL_DIR, "config.json");
		if (!Files.exists(modelPath) || !Files.exists(configPath)) {
			System.out.println("Model files not found, skipping");
			return;
		}

		KokoroTtsConfig config = KokoroTtsConfig.builder()
			.modelPath(modelPath.toString())
			.voicesDir(voicesDir.toString())
			.configPath(configPath.toString())
			.defaultVoice("zf_001")
			.defaultSpeed(1.0f)
			.build();

		String phonemes = "ㄋㄧ ㄏㄠ ㄨㄛ ㄕ";
		try (KokoroTts tts = new KokoroTts(config)) {

			// 1) 从 TTS 内部复刻出真实的 token + ref_s
			var vocab = (net.dreamlu.mica.ai.tts.engine.Vocab) getField(tts, "vocab");
			var voiceManager = (net.dreamlu.mica.ai.tts.engine.VoiceManager) getField(tts, "voiceManager");
			String filtered = vocab.filter(phonemes);
			List<Integer> tokenList = vocab.tokenize(filtered);
			long[] inputIds = new long[tokenList.size() + 2];
			inputIds[0] = 0;
			for (int i = 0; i < tokenList.size(); i++) inputIds[i + 1] = tokenList.get(i);
			inputIds[inputIds.length - 1] = 0;
			float[] refS = voiceManager.getStyle("zf_001", inputIds.length);

			System.out.println("phonemes   = '" + phonemes + "' (len=" + phonemes.length() + ")");
			System.out.println("filtered   = '" + filtered + "' (len=" + filtered.length() + ")");
			System.out.println("tokenList  = " + tokenList + "  (size=" + tokenList.size() + ")");
			StringBuilder sb = new StringBuilder("input_ids = [");
			for (int i = 0; i < inputIds.length; i++) {
				if (i > 0) sb.append(", ");
				sb.append(inputIds[i]);
			}
			sb.append("]  len=" + inputIds.length);
			System.out.println(sb);
			System.out.println("ref_s      first 5 = " + first5(refS));
			System.out.println("ref_s      last 5  = " + last5(refS));
			float rmin = Float.POSITIVE_INFINITY, rmax = Float.NEGATIVE_INFINITY, rs = 0;
			for (float v : refS) { if (v < rmin) rmin = v; if (v > rmax) rmax = v; rs += v; }
			System.out.printf("ref_s      min=%.4f  max=%.4f  mean=%.4f%n", rmin, rmax, rs / refS.length);

			// 2) 落盘 input_ids / ref_s / speed
			Path idsOut = Path.of(OUT_DIR, "java_input_ids.bin");
			Path refOut = Path.of(OUT_DIR, "java_ref_s.bin");
			Path spdOut = Path.of(OUT_DIR, "java_speed.txt");
			Files.write(idsOut, longArrayToBytes(inputIds));
			Files.write(refOut, floatArrayToBytes(refS));
			Files.writeString(spdOut, "1.0");
			System.out.println("Wrote " + idsOut + "  " + refOut + "  " + spdOut);

			// 3) 跑一次完整 TTS 并保存 WAV（已包含 trimSilence）
			TtsResult result = tts.synthesizeFromPhonemes(phonemes, "zf_001", 1.0f);
			System.out.printf("TTS(trim) audio_len=%d  mean_abs=%.4f%n",
				result.audio().length, meanAbs(result.audio()));
			System.out.println("TTS(trim) first 10 = " + first10(result.audio()));
			Path wav = Path.of(OUT_DIR, "probe_java_dump.wav");
			saveWav(result.audio(), 24000, wav);
			System.out.println("Wrote " + wav);

			// 4) 同样输入直接调 ONNX（不走 trimSilence），看原始输出
			float[] raw = runRawInference(inputIds, refS, 1.0f);
			System.out.printf("RAW(no-trim) audio_len=%d  mean_abs=%.4f%n",
				raw.length, meanAbs(raw));
			System.out.println("RAW(no-trim) first 30 = " + firstN(raw, 30));
			System.out.println("RAW(no-trim) 10  = " + first10(raw));
			Path rawWav = Path.of(OUT_DIR, "probe_java_dump_raw.wav");
			saveWav(raw, 24000, rawWav);
			System.out.println("Wrote " + rawWav);

			// 5) 再跑一遍确定 ORT 1.26.0 是 deterministic
			float[] raw2 = runRawInference(inputIds, refS, 1.0f);
			float diff = 0;
			for (int i = 0; i < Math.min(raw.length, raw2.length); i++) diff += Math.abs(raw[i] - raw2[i]);
			System.out.printf("Determinism: |raw - raw2|.sum = %.6f  (0=完全一致)%n", diff);

			// 6) 复用 KokoroEngine.inference 同样输入看是否一致
			var engine = (net.dreamlu.mica.ai.tts.engine.KokoroEngine) getField(tts, "engine");
			float[] viaEngine = engine.inference(inputIds, refS, 1.0f);
			float diff2 = 0;
			for (int i = 0; i < Math.min(raw.length, viaEngine.length); i++) diff2 += Math.abs(raw[i] - viaEngine[i]);
			System.out.printf("raw vs engine.inference: |diff|.sum = %.6f  len(raw)=%d  len(engine)=%d%n",
				diff2, raw.length, viaEngine.length);
			System.out.println("viaEngine first 10 = " + first10(viaEngine));
			System.out.println("viaEngine first 30 = " + firstN(viaEngine, 30));

			// 7) 直接调用 engine.inference 后跑 trimSilence 看是否会变 0.0030
			float[] engineTrimmed = net.dreamlu.mica.ai.tts.engine.TextFrontend.trimSilence(viaEngine, 0.01f);
			System.out.printf("viaEngine + trimSilence: len=%d  mean_abs=%.4f  first 10 = %s%n",
				engineTrimmed.length, meanAbs(engineTrimmed), first10(engineTrimmed));

			// 8) 拿到 KokoroEngine 内部 session 自己重跑（仅验 session 本身）
			var sessField = engine.getClass().getDeclaredField("session");
			sessField.setAccessible(true);
			ai.onnxruntime.OrtSession sharedSess = (ai.onnxruntime.OrtSession) sessField.get(engine);
			LongBuffer ib3 = LongBuffer.wrap(inputIds);
			OnnxTensor idsT3 = OnnxTensor.createTensor(ai.onnxruntime.OrtEnvironment.getEnvironment(), ib3, new long[]{1, inputIds.length});
			FloatBuffer rb3 = FloatBuffer.wrap(refS);
			OnnxTensor refT3 = OnnxTensor.createTensor(ai.onnxruntime.OrtEnvironment.getEnvironment(), rb3, new long[]{1, refS.length});
			FloatBuffer sb3 = FloatBuffer.wrap(new float[]{1.0f});
			OnnxTensor spdT3 = OnnxTensor.createTensor(ai.onnxruntime.OrtEnvironment.getEnvironment(), sb3, new long[]{1});
			java.util.Map<String, OnnxTensor> inp = new java.util.LinkedHashMap<>();
			inp.put("input_ids", idsT3); inp.put("ref_s", refT3); inp.put("speed", spdT3);
			try (ai.onnxruntime.OrtSession.Result r = sharedSess.run(inp)) {
				OnnxTensor a = (OnnxTensor) r.get("audio").orElseThrow();
				float[] viaShared = a.getFloatBuffer().array();
				float diff3 = 0;
				for (int i = 0; i < Math.min(raw.length, viaShared.length); i++) diff3 += Math.abs(raw[i] - viaShared[i]);
				System.out.printf("raw vs sharedSess.run: |diff|.sum = %.6f  len(viaShared)=%d%n", diff3, viaShared.length);
				System.out.printf("viaShared mean_abs=%.4f  min=%.4f  max=%.4f  len=%d%n",
					meanAbs(viaShared), min(viaShared), max(viaShared), viaShared.length);
				System.out.println("viaShared first 10 (raw) = " + first10F(viaShared));
				System.out.println("viaShared first 10 (high precision) = " + firstN(viaShared, 10));

				// 9) 同一 sharedSess 连跑 2 次，看是否 deterministic
				LongBuffer ib4 = LongBuffer.wrap(inputIds);
				OnnxTensor idsT4 = OnnxTensor.createTensor(ai.onnxruntime.OrtEnvironment.getEnvironment(), ib4, new long[]{1, inputIds.length});
				FloatBuffer rb4 = FloatBuffer.wrap(refS);
				OnnxTensor refT4 = OnnxTensor.createTensor(ai.onnxruntime.OrtEnvironment.getEnvironment(), rb4, new long[]{1, refS.length});
				FloatBuffer sb4 = FloatBuffer.wrap(new float[]{1.0f});
				OnnxTensor spdT4 = OnnxTensor.createTensor(ai.onnxruntime.OrtEnvironment.getEnvironment(), sb4, new long[]{1});
				java.util.Map<String, OnnxTensor> inp2 = new java.util.LinkedHashMap<>();
				inp2.put("input_ids", idsT4); inp2.put("ref_s", refT4); inp2.put("speed", spdT4);
				try (ai.onnxruntime.OrtSession.Result r2 = sharedSess.run(inp2)) {
					OnnxTensor a2 = (OnnxTensor) r2.get("audio").orElseThrow();
					float[] viaShared2 = a2.getFloatBuffer().array();
					float diff4 = 0;
					for (int i = 0; i < Math.min(viaShared.length, viaShared2.length); i++) diff4 += Math.abs(viaShared[i] - viaShared2[i]);
					System.out.printf("sharedSess x2: |diff|.sum = %.6f  viaShared2 mean_abs=%.4f%n", diff4, meanAbs(viaShared2));
				}
				// 10) hypothesis: getFloatBuffer() 拿到的是 internal pool buffer，会被下次 run 覆盖
				// 验证：不立即 .array() 复制，而是先跑第二次再取第一个，会被覆盖吗？
				OnnxTensor aa = (OnnxTensor) sharedSess.run(inp).get("audio").orElseThrow();
				java.nio.FloatBuffer fb1 = aa.getFloatBuffer();
				// 立即 run 第二次
				ai.onnxruntime.OrtSession.Result rr2 = sharedSess.run(inp);
				OnnxTensor bb = (OnnxTensor) rr2.get("audio").orElseThrow();
				java.nio.FloatBuffer fb2 = bb.getFloatBuffer();
				// 现在再读 fb1（取出来拷贝到 heap）
				float[] lateCopy = new float[fb1.capacity()];
				((java.nio.Buffer) fb1).position(0);
				fb1.get(lateCopy);
				// fb2 同样拷贝
				float[] fb2copy = new float[fb2.capacity()];
				((java.nio.Buffer) fb2).position(0);
				fb2.get(fb2copy);
				float diffLate = 0;
				for (int i = 0; i < Math.min(fb2copy.length, lateCopy.length); i++) diffLate += Math.abs(fb2copy[i] - lateCopy[i]);
				System.out.printf("late-copy after 2nd run: |fb1_late - fb2|.sum = %.6f  (如果 0 则 buffer 被覆盖)%n", diffLate);
							System.out.printf("  fb1_late first 5 (e): %s%n", first5e(lateCopy));
				System.out.printf("  fb2 first 5 (e):      %s%n", first5e(fb2copy));
				aa.close(); bb.close();
				idsT3.close(); refT3.close(); spdT3.close();
				idsT4.close(); refT4.close(); spdT4.close();
			}

			// 11) 关键实验：fresh session 跑第 1/2/3 次，看是否 1st run 异常
			System.out.println("\n=== 11) 1st / 2nd / 3rd run with fresh session ===");
			float[] firstRun = runFreshSessionNthRun(1, inputIds, refS, 1.0f);
			System.out.printf("  1st run: len=%d  mean_abs=%.6f  first 10 (hp) = %s%n",
				firstRun.length, meanAbs(firstRun), first10F(firstRun));
			System.out.printf("  1st run: 前 110 帧 (trim 范围) = %s%n", firstN(firstRun, 110));
			float[] secondRun = runFreshSessionNthRun(2, inputIds, refS, 1.0f);
			System.out.printf("  2nd run: len=%d  mean_abs=%.6f  first 10 (hp) = %s%n",
				secondRun.length, meanAbs(secondRun), first10F(secondRun));
			System.out.printf("  2nd run: 前 110 帧 (trim 范围) = %s%n", firstN(secondRun, 110));
			float[] thirdRun = runFreshSessionNthRun(3, inputIds, refS, 1.0f);
			System.out.printf("  3rd run: len=%d  mean_abs=%.6f  first 10 (hp) = %s%n",
				thirdRun.length, meanAbs(thirdRun), first10F(thirdRun));
			System.out.printf("  3rd run: 前 110 帧 (trim 范围) = %s%n", firstN(thirdRun, 110));
			float diff12 = 0, diff13 = 0, diff23 = 0;
			for (int i = 0; i < Math.min(firstRun.length, secondRun.length); i++) diff12 += Math.abs(firstRun[i] - secondRun[i]);
			for (int i = 0; i < Math.min(firstRun.length, thirdRun.length); i++) diff13 += Math.abs(firstRun[i] - thirdRun[i]);
			for (int i = 0; i < Math.min(secondRun.length, thirdRun.length); i++) diff23 += Math.abs(secondRun[i] - thirdRun[i]);
			System.out.printf("  |1st - 2nd|.sum = %.6f%n", diff12);
			System.out.printf("  |1st - 3rd|.sum = %.6f%n", diff13);
			System.out.printf("  |2nd - 3rd|.sum = %.6f%n", diff23);
			// 前 110 帧的 mean_abs（trim 范围）
			double first110 = 0, second110 = 0, third110 = 0;
			for (int i = 0; i < 110; i++) { first110 += Math.abs(firstRun[i]); second110 += Math.abs(secondRun[i]); third110 += Math.abs(thirdRun[i]); }
			System.out.printf("  前 110 帧 mean_abs: 1st=%.6f  2nd=%.6f  3rd=%.6f%n", first110/110, second110/110, third110/110);

			// 12) 1st run 跑 trimSilence，看 trimmed 的前 110 帧是什么
			float[] trimmed = net.dreamlu.mica.ai.tts.engine.TextFrontend.trimSilence(firstRun, 0.01f);
			System.out.printf("trimSilence(1st, 0.01): len=%d  mean_abs=%.6f%n", trimmed.length, meanAbs(trimmed));
			System.out.printf("  trimmed first 10 (hp) = %s%n", first10F(trimmed));
			System.out.printf("  trimmed first 110 (4dec) = %s%n", firstN(trimmed, 110));
			// 同样的，2nd/3rd run trim 后输出
			float[] trimmed2 = net.dreamlu.mica.ai.tts.engine.TextFrontend.trimSilence(secondRun, 0.01f);
			float[] trimmed3 = net.dreamlu.mica.ai.tts.engine.TextFrontend.trimSilence(thirdRun, 0.01f);
			System.out.printf("trimSilence(2nd, 0.01): len=%d  mean_abs=%.6f  first 10 (hp) = %s%n",
				trimmed2.length, meanAbs(trimmed2), first10F(trimmed2));
			System.out.printf("trimSilence(3rd, 0.01): len=%d  mean_abs=%.6f  first 10 (hp) = %s%n",
				trimmed3.length, meanAbs(trimmed3), first10F(trimmed3));

			// 13) 看 1st run 完整 firstRun 在 100~300 区间的值
			System.out.printf("1st run 0~50 (4dec): %s%n", sliceN(firstRun, 0, 50));
			System.out.printf("1st run 100~300 (4dec): %s%n", sliceN(firstRun, 100, 300));
			System.out.printf("1st run 300~500 (hp): %s%n", sliceN(firstRun, 300, 500));
			// 找 firstRun 第一个 |x| >= 0.01 的索引
			int firstNonSilent = -1;
			for (int i = 0; i < firstRun.length; i++) {
				if (Math.abs(firstRun[i]) >= 0.01f) { firstNonSilent = i; break; }
			}
			System.out.printf("1st run: 第一个 |x|>=0.01 的索引 = %d, 该位置值 = %+.6e%n",
				firstNonSilent, firstNonSilent >= 0 ? firstRun[firstNonSilent] : 0f);
			// 找 last non-silent
			int lastNonSilent = -1;
			for (int i = firstRun.length - 1; i >= 0; i--) {
				if (Math.abs(firstRun[i]) >= 0.01f) { lastNonSilent = i; break; }
			}
			System.out.printf("1st run: 最后一个 |x|>=0.01 的索引 = %d, 该位置值 = %+.6e%n",
				lastNonSilent, lastNonSilent >= 0 ? firstRun[lastNonSilent] : 0f);
		}
	}

	private static String sliceN(float[] a, int from, int to) {
		StringBuilder sb = new StringBuilder("[");
		for (int i = from; i < Math.min(to, a.length); i++) {
			if (i > from) sb.append(", ");
			if ((i - from) % 10 == 0 && i > from) sb.append("\n  [" + i + "] ");
			sb.append(String.format("%+.4f", a[i]));
		}
		return sb.append("]").toString();
	}

	private static String first5e(float[] a) {
		StringBuilder sb = new StringBuilder("[");
		for (int i = 0; i < Math.min(5, a.length); i++) { if (i > 0) sb.append(", "); sb.append(String.format("%.6e", a[i])); }
		return sb.append("]").toString();
	}

	private static float min(float[] a) {
		float m = Float.POSITIVE_INFINITY;
		for (float v : a) if (v < m) m = v;
		return m;
	}

	private static float max(float[] a) {
		float m = Float.NEGATIVE_INFINITY;
		for (float v : a) if (v > m) m = v;
		return m;
	}

	private static String first10F(float[] a) {
		StringBuilder sb = new StringBuilder("[");
		for (int i = 0; i < Math.min(10, a.length); i++) {
			if (i > 0) sb.append(", ");
			sb.append(String.format("%.6e", a[i]));
		}
		return sb.append("]").toString();
	}

	private static float[] runRawInference(long[] inputIds, float[] refS, float speed) throws Exception {
		var env = OrtEnvironment.getEnvironment();
		var session = env.createSession("E:\\codes\\ai\\kokoro-onnx\\model\\model_dynamic.onnx",
			new OrtSession.SessionOptions());
		try {
			LongBuffer ib = LongBuffer.wrap(inputIds);
			OnnxTensor idsT = OnnxTensor.createTensor(env, ib, new long[]{1, inputIds.length});
			FloatBuffer rb = FloatBuffer.wrap(refS);
			OnnxTensor refT = OnnxTensor.createTensor(env, rb, new long[]{1, refS.length});
			FloatBuffer sb2 = FloatBuffer.wrap(new float[]{speed});
			OnnxTensor spdT = OnnxTensor.createTensor(env, sb2, new long[]{1});
			Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
			inputs.put("input_ids", idsT);
			inputs.put("ref_s", refT);
			inputs.put("speed", spdT);
			try (OrtSession.Result r = session.run(inputs)) {
				OnnxTensor a = (OnnxTensor) r.get("audio").orElseThrow();
				return a.getFloatBuffer().array();
			} finally {
				idsT.close(); refT.close(); spdT.close();
			}
		} finally {
			session.close();
		}
	}

	/** 和 KokoroEngine.inference 写法一模一样，但用 new session 并接收 runIndex 用于日志 */
	private static float[] runFreshSessionNthRun(int runIndex, long[] inputIds, float[] refS, float speed) throws Exception {
		var env = OrtEnvironment.getEnvironment();
		var session = env.createSession("E:\\codes\\ai\\kokoro-onnx\\model\\model_dynamic.onnx",
			new OrtSession.SessionOptions());
		try {
			// 先做 runIndex-1 次 dummy run（不读结果，只为占位）
			for (int i = 0; i < runIndex - 1; i++) {
				LongBuffer ib = LongBuffer.wrap(inputIds);
				OnnxTensor idsT = OnnxTensor.createTensor(env, ib, new long[]{1, inputIds.length});
				FloatBuffer rb = FloatBuffer.wrap(refS);
				OnnxTensor refT = OnnxTensor.createTensor(env, rb, new long[]{1, refS.length});
				FloatBuffer sb2 = FloatBuffer.wrap(new float[]{speed});
				OnnxTensor spdT = OnnxTensor.createTensor(env, sb2, new long[]{1});
				Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
				inputs.put("input_ids", idsT);
				inputs.put("ref_s", refT);
				inputs.put("speed", spdT);
				try (OrtSession.Result r = session.run(inputs)) {
					OnnxTensor a = (OnnxTensor) r.get("audio").orElseThrow();
					a.getFloatBuffer().array();
				} finally { idsT.close(); refT.close(); spdT.close(); }
			}
			// 第 runIndex 次（要记录）
			LongBuffer ib = LongBuffer.wrap(inputIds);
			OnnxTensor idsT = OnnxTensor.createTensor(env, ib, new long[]{1, inputIds.length});
			FloatBuffer rb = FloatBuffer.wrap(refS);
			OnnxTensor refT = OnnxTensor.createTensor(env, rb, new long[]{1, refS.length});
			FloatBuffer sb2 = FloatBuffer.wrap(new float[]{speed});
			OnnxTensor spdT = OnnxTensor.createTensor(env, sb2, new long[]{1});
			Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
			inputs.put("input_ids", idsT);
			inputs.put("ref_s", refT);
			inputs.put("speed", spdT);
			try (OrtSession.Result r = session.run(inputs)) {
				OnnxTensor a = (OnnxTensor) r.get("audio").orElseThrow();
				return a.getFloatBuffer().array();
			} finally { idsT.close(); refT.close(); spdT.close(); }
		} finally {
			session.close();
		}
	}

	private static String firstN(float[] a, int n) {
		StringBuilder sb = new StringBuilder("[");
		for (int i = 0; i < Math.min(n, a.length); i++) {
			if (i > 0) sb.append(", ");
			sb.append(String.format("%+.4f", a[i]));
		}
		return sb.append("]").toString();
	}

	private static Object getField(Object obj, String name) throws Exception {
		var f = obj.getClass().getDeclaredField(name);
		f.setAccessible(true);
		return f.get(obj);
	}

	private static String first5(float[] a) {
		StringBuilder sb = new StringBuilder("[");
		for (int i = 0; i < Math.min(5, a.length); i++) { if (i > 0) sb.append(", "); sb.append(a[i]); }
		return sb.append("]").toString();
	}

	private static String last5(float[] a) {
		StringBuilder sb = new StringBuilder("[");
		int start = Math.max(0, a.length - 5);
		for (int i = start; i < a.length; i++) { if (i > start) sb.append(", "); sb.append(a[i]); }
		return sb.append("]").toString();
	}

	private static String first10(float[] a) {
		StringBuilder sb = new StringBuilder("[");
		for (int i = 0; i < Math.min(10, a.length); i++) { if (i > 0) sb.append(", "); sb.append(String.format("%+.4f", a[i])); }
		return sb.append("]").toString();
	}

	private static float meanAbs(float[] a) {
		double s = 0;
		for (float v : a) s += Math.abs(v);
		return (float) (s / a.length);
	}

	private static byte[] longArrayToBytes(long[] arr) {
		ByteBuffer bb = ByteBuffer.allocate(arr.length * 8).order(ByteOrder.LITTLE_ENDIAN);
		bb.asLongBuffer().put(arr);
		return bb.array();
	}

	private static byte[] floatArrayToBytes(float[] arr) {
		ByteBuffer bb = ByteBuffer.allocate(arr.length * 4).order(ByteOrder.LITTLE_ENDIAN);
		bb.asFloatBuffer().put(arr);
		return bb.array();
	}

	private static void saveWav(float[] audio, int sampleRate, Path file) throws IOException {
		try (OutputStream os = Files.newOutputStream(file);
			 DataOutputStream dos = new DataOutputStream(os)) {
			dos.writeBytes("RIFF");
			dos.writeInt(Integer.reverseBytes(36 + audio.length * 2));
			dos.writeBytes("WAVE");
			dos.writeBytes("fmt ");
			dos.writeInt(Integer.reverseBytes(16));
			dos.writeShort(Short.reverseBytes((short) 1));
			dos.writeShort(Short.reverseBytes((short) 1));
			dos.writeInt(Integer.reverseBytes(sampleRate));
			dos.writeInt(Integer.reverseBytes(sampleRate * 2));
			dos.writeShort(Short.reverseBytes((short) 2));
			dos.writeShort(Short.reverseBytes((short) 16));
			dos.writeBytes("data");
			dos.writeInt(Integer.reverseBytes(audio.length * 2));
			for (float sample : audio) {
				float c = Math.max(-1f, Math.min(1f, sample));
				short pcm = (short) (c * 32767f);
				dos.writeShort(Short.reverseBytes(pcm));
			}
		}
	}
}
