/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.speaker;

import net.dreamlu.mica.ai.common.utils.AudioUtils;
import net.dreamlu.mica.ai.speaker.internal.FBankExtractor;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SpeakerVerifier 真实语音集成测试。
 *
 * <p>使用外部目录中的真实 ONNX 模型和 WAV 录音进行端到端测试。
 * 文件不存在时自动跳过（不影响 CI）。
 *
 * <p>预期目录结构（参考 mica-ai-voice 的 SenseVoiceTest 模式）：
 * <pre>{@code
 * E:\codes\ai\Speaker-ONNX\
 * ├── model\
 * │   └── speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
 * └── audio\
 *     ├── speaker1_enroll_1.wav    # 说话人 A 注册音频 1
 *     ├── speaker1_enroll_2.wav    # 说话人 A 注册音频 2
 *     ├── speaker1_enroll_3.wav    # 说话人 A 注册音频 3
 *     ├── speaker1_test.wav        # 说话人 A 测试音频（应匹配）
 *     ├── speaker2_enroll_1.wav    # 说话人 B 注册音频
 *     └── speaker2_test.wav        # 说话人 B 测试音频（不应匹配 A）
 * }</pre>
 *
 * <p>模型下载：
 * <a href="https://github.com/alibaba-damo-academy/3D-Speaker">3D-Speaker</a>
 * → speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
 */
@DisplayName("SpeakerVerifier 真实语音集成测试")
class SpeakerVerifierIntegrationTest {

	// ====== 路径配置（指向 Speaker-ONNX 项目目录） ======
	private static final String SV_ONNX_DIR = "E:\\codes\\ai\\Speaker-ONNX";
	private static final String MODEL_DIR = SV_ONNX_DIR + "\\model";
	private static final String MODEL_PATH = MODEL_DIR + "\\speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
	private static final String AUDIO_DIR = SV_ONNX_DIR + "\\audio";

	// 说话人 A 音频路径
	private static final String SPK1_ENROLL_1 = AUDIO_DIR + "\\speaker1_enroll_1.wav";
	private static final String SPK1_ENROLL_2 = AUDIO_DIR + "\\speaker1_enroll_2.wav";
	private static final String SPK1_ENROLL_3 = AUDIO_DIR + "\\speaker1_enroll_3.wav";
	private static final String SPK1_TEST = AUDIO_DIR + "\\speaker1_test.wav";

	// 说话人 B 音频路径
	private static final String SPK2_ENROLL_1 = AUDIO_DIR + "\\speaker2_enroll_1.wav";
	private static final String SPK2_TEST = AUDIO_DIR + "\\speaker2_test.wav";

	// ====== 跳过检查工具 ======

	/**
	 * 检查模型文件是否存在，不存在则用 assumeTrue 跳过测试（在报告中标记为 SKIPPED）。
	 */
	private void assumeModelExists() {
		boolean exists = Files.isRegularFile(Path.of(MODEL_PATH));
		if (!exists) {
			System.out.println("⚠️ 模型文件不存在，跳过推理测试: " + MODEL_PATH);
			System.out.println("   下载地址: https://github.com/modelscope/3D-Speaker");
		}
		Assumptions.assumeTrue(exists, "模型文件不存在: " + MODEL_PATH);
	}

	/**
	 * 检查指定音频文件是否存在，不存在则用 assumeTrue 跳过测试。
	 */
	private void assumeAudioExists(String path) {
		boolean exists = Files.isRegularFile(Path.of(path));
		if (!exists) {
			System.out.println("⚠️ 音频文件不存在，跳过测试: " + path);
		}
		Assumptions.assumeTrue(exists, "音频文件不存在: " + path);
	}

	/**
	 * 检查所有指定音频文件是否存在。
	 */
	private void assumeAllAudioExists(String... paths) {
		for (String p : paths) {
			assumeAudioExists(p);
		}
	}

	// ====== 真实音频预处理测试（不依赖 ONNX 模型） ======

	@Test
	@DisplayName("真实 WAV 加载：AudioUtils 能正确读取 16kHz/16bit/单声道 PCM")
	void testLoadRealWav() throws IOException {
		assumeAudioExists(SPK1_ENROLL_1);

		float[] audio = AudioUtils.loadWavAsFloat(Path.of(SPK1_ENROLL_1));
		assertNotNull(audio, "加载的音频数据不应为 null");
		assertTrue(audio.length > 0, "音频长度应 > 0");

		double durationSec = audio.length / 16000.0;
		System.out.printf("[Audio] 采样点数: %d, 时长: %.2fs%n", audio.length, durationSec);

		// 真实语音应在 0.5s ~ 30s 之间
		assertTrue(durationSec >= 0.5, "音频过短 (<0.5s)");
		assertTrue(durationSec <= 30, "音频过长 (>30s)");

		// 真实语音应有非零能量
		double energy = 0;
		for (float v : audio) energy += (double) v * v;
		energy = Math.sqrt(energy / audio.length);
		assertTrue(energy > 1e-4, "音频能量过低 (RMS=" + energy + ")，可能为静音");
		System.out.printf("[Audio] RMS 能量: %.6f%n", energy);
	}

	@Test
	@DisplayName("真实语音 FBank 特征：维度 [T, 80]，数值合理")
	void testRealFbankExtraction() throws IOException {
		assumeAudioExists(SPK1_ENROLL_1);

		float[] audio = AudioUtils.loadWavAsFloat(Path.of(SPK1_ENROLL_1));
		FBankExtractor frontend = new FBankExtractor();
		float[][] feats = frontend.extract(audio);

		assertNotNull(feats, "FBank 特征不应为 null");
		assertTrue(feats.length > 0, "帧数应 > 0");
		assertEquals(80, feats[0].length, "Mel 频带数应为 80");

		int T = feats.length;
		double durationSec = audio.length / 16000.0;
		int expectedFrames = (int) (durationSec * 100); // 10ms 帧移 ≈ 100 帧/秒
		System.out.printf("[FBank] 帧数: %d (预期约 %d), Mel 频带: 80%n", T, expectedFrames);

		// 帧数应在预期值的 ±20% 范围内
		assertTrue(Math.abs(T - expectedFrames) < expectedFrames * 0.2 + 10,
			"帧数 " + T + " 偏离预期 " + expectedFrames + " 过多");

		// 检查特征值范围（log Mel 通常在 -20 ~ +10 之间）
		double min = Double.MAX_VALUE, max = -Double.MAX_VALUE, sum = 0;
		for (float[] frame : feats) {
			for (float v : frame) {
				if (v < min) min = v;
				if (v > max) max = v;
				sum += v;
			}
		}
		double mean = sum / (T * 80);
		System.out.printf("[FBank] 数值范围: min=%.2f, max=%.2f, mean=%.2f%n", min, max, mean);
		assertTrue(min > -30, "FBank min 异常低: " + min);
		assertTrue(max < 30, "FBank max 异常高: " + max);
	}

	// ====== ONNX 推理测试（依赖模型文件） ======

	@Test
	@DisplayName("Embedding 提取：维度 192，L2 范数 ≈ 1")
	void testExtractEmbedding() throws IOException {
		assumeModelExists();
		assumeAudioExists(SPK1_ENROLL_1);

		try (SpeakerVerifier verifier = new SpeakerVerifier(MODEL_PATH)) {
			long t0 = System.currentTimeMillis();
			float[] emb = verifier.extractEmbedding(Path.of(SPK1_ENROLL_1));
			long elapsed = System.currentTimeMillis() - t0;

			assertNotNull(emb, "embedding 不应为 null");
			assertEquals(SpeakerVerifier.EMBEDDING_DIM, emb.length,
				"embedding 维度应为 " + SpeakerVerifier.EMBEDDING_DIM);

			// L2 范数应 ≈ 1（已归一化）
			double norm = 0;
			for (float v : emb) norm += (double) v * v;
			norm = Math.sqrt(norm);
			assertEquals(1.0, norm, 1e-3, "L2 范数应 ≈ 1");

			System.out.printf("[Embedding] 维度: %d, L2 范数: %.6f, 耗时: %dms%n",
				emb.length, norm, elapsed);

			// 打印前 10 维预览
			System.out.print("[Embedding] 前 10 维: [");
			for (int i = 0; i < 10 && i < emb.length; i++) {
				System.out.printf("%.4f%s", emb[i], i < 9 ? ", " : "");
			}
			System.out.println("]");
		}
	}

	@Test
	@DisplayName("同一说话人不同录音：余弦相似度应 > 阈值")
	void testSameSpeakerHighSimilarity() throws IOException {
		assumeModelExists();
		assumeAllAudioExists(SPK1_ENROLL_1, SPK1_TEST);

		try (SpeakerVerifier verifier = new SpeakerVerifier(MODEL_PATH)) {
			float[] emb1 = verifier.extractEmbedding(Path.of(SPK1_ENROLL_1));
			float[] emb2 = verifier.extractEmbedding(Path.of(SPK1_TEST));

			float score = SpeakerVerifier.cosineSimilarity(emb1, emb2);
			System.out.printf("[同说话人] cos 相似度: %.4f (阈值 %.2f)%n",
				score, SpeakerVerifier.DEFAULT_THRESHOLD);

			assertTrue(score > SpeakerVerifier.DEFAULT_THRESHOLD,
				"同一说话人相似度 " + score + " 应 > 阈值 " + SpeakerVerifier.DEFAULT_THRESHOLD);
		}
	}

	@Test
	@DisplayName("不同说话人：余弦相似度应 < 阈值")
	void testDifferentSpeakerLowSimilarity() throws IOException {
		assumeModelExists();
		assumeAllAudioExists(SPK1_TEST, SPK2_TEST);

		try (SpeakerVerifier verifier = new SpeakerVerifier(MODEL_PATH)) {
			float[] emb1 = verifier.extractEmbedding(Path.of(SPK1_TEST));
			float[] emb2 = verifier.extractEmbedding(Path.of(SPK2_TEST));

			float score = SpeakerVerifier.cosineSimilarity(emb1, emb2);
			System.out.printf("[不同说话人] cos 相似度: %.4f (阈值 %.2f)%n",
				score, SpeakerVerifier.DEFAULT_THRESHOLD);

			assertTrue(score < SpeakerVerifier.DEFAULT_THRESHOLD,
				"不同说话人相似度 " + score + " 应 < 阈值 " + SpeakerVerifier.DEFAULT_THRESHOLD);
		}
	}

	@Test
	@DisplayName("声纹注册：多段音频注册后与测试音频匹配")
	void testEnrollSpeaker() throws IOException {
		assumeModelExists();
		assumeAllAudioExists(SPK1_ENROLL_1, SPK1_ENROLL_2, SPK1_ENROLL_3, SPK1_TEST);

		try (SpeakerVerifier verifier = new SpeakerVerifier(MODEL_PATH)) {
			// 用 3 段音频注册
			float[] enrolled = verifier.enrollSpeaker(List.of(
				Path.of(SPK1_ENROLL_1),
				Path.of(SPK1_ENROLL_2),
				Path.of(SPK1_ENROLL_3)
			));

			assertEquals(SpeakerVerifier.EMBEDDING_DIM, enrolled.length);

			// 注册声纹 L2 范数 ≈ 1
			double norm = 0;
			for (float v : enrolled) norm += (double) v * v;
			assertEquals(1.0, Math.sqrt(norm), 1e-3, "注册声纹应已 L2 归一化");

			// 与测试音频比对
			float[] testEmb = verifier.extractEmbedding(Path.of(SPK1_TEST));
			float score = SpeakerVerifier.cosineSimilarity(enrolled, testEmb);

			System.out.printf("[注册验证] 注册声纹 vs 测试音频 cos: %.4f%n", score);
			assertTrue(score > SpeakerVerifier.DEFAULT_THRESHOLD,
				"注册声纹与本人测试音频相似度 " + score + " 应 > 阈值");

			// verify() 方法测试
			boolean matched = verifier.verify(enrolled, Path.of(SPK1_TEST));
			assertTrue(matched, "verify() 应返回 true");
			System.out.println("[注册验证] verify() 返回: true ✓");
		}
	}

	@Test
	@DisplayName("verify() 方法：同一人匹配 / 不同人不匹配")
	void testVerifyMethod() throws IOException {
		assumeModelExists();
		assumeAllAudioExists(SPK1_ENROLL_1, SPK1_TEST, SPK2_TEST);

		try (SpeakerVerifier verifier = new SpeakerVerifier(MODEL_PATH)) {
			float[] enrolled = verifier.extractEmbedding(Path.of(SPK1_ENROLL_1));

			// 同一人：verify 应返回 true
			boolean selfMatch = verifier.verify(enrolled, Path.of(SPK1_TEST));
			System.out.printf("[verify] 本人测试: %s%n", selfMatch ? "匹配 ✓" : "不匹配 ✗");
			assertTrue(selfMatch, "本人测试应匹配");

			// 不同人：verify 应返回 false
			boolean otherMatch = verifier.verify(enrolled, Path.of(SPK2_TEST));
			System.out.printf("[verify] 他人测试: %s%n", otherMatch ? "匹配 ✗" : "不匹配 ✓");
			assertFalse(otherMatch, "他人测试不应匹配");

			// 自定义阈值测试
			boolean strictMatch = verifier.verify(enrolled, Path.of(SPK1_TEST), 0.7f);
			System.out.printf("[verify] 阈值 0.7 本人: %s%n", strictMatch ? "匹配 ✓" : "不匹配");
		}
	}

	@Test
	@DisplayName("性能测速：3 轮推理平均耗时")
	void testPerformanceBenchmark() throws IOException {
		assumeModelExists();
		assumeAudioExists(SPK1_ENROLL_1);

		try (SpeakerVerifier verifier = new SpeakerVerifier(MODEL_PATH)) {
			Path wav = Path.of(SPK1_ENROLL_1);
			float[] audio = AudioUtils.loadWavAsFloat(wav);
			double durationSec = audio.length / 16000.0;

			System.out.printf("[Perf] 音频时长: %.2fs%n", durationSec);
			System.out.println("[Perf] 开始 3 轮推理测速...");

			// 第 1 轮（含模型预热）
			long t0 = System.currentTimeMillis();
			verifier.extractEmbedding(audio);
			long firstRun = System.currentTimeMillis() - t0;

			// 第 2、3 轮（稳态）
			long total = 0;
			for (int i = 2; i <= 3; i++) {
				t0 = System.currentTimeMillis();
				verifier.extractEmbedding(audio);
				long elapsed = System.currentTimeMillis() - t0;
				total += elapsed;
				System.out.printf(" >>> 第 %d 轮耗时: %dms (RTF: %.2f)%n",
					i, elapsed, elapsed / 1000.0 / durationSec);
			}

			System.out.printf("[Perf] 首轮（含预热）: %dms | 稳态平均: %dms | RTF: %.3f%n",
				firstRun, total / 2, (total / 2) / 1000.0 / durationSec);

			assertTrue(total / 2 < 5000, "稳态推理耗时过长 (>5s)");
		}
	}

	@Test
	@DisplayName("同一音频重复提取 embedding 一致性")
	void testReproducibility() throws IOException {
		assumeModelExists();
		assumeAudioExists(SPK1_ENROLL_1);

		try (SpeakerVerifier verifier = new SpeakerVerifier(MODEL_PATH)) {
			float[] emb1 = verifier.extractEmbedding(Path.of(SPK1_ENROLL_1));
			float[] emb2 = verifier.extractEmbedding(Path.of(SPK1_ENROLL_1));

			float score = SpeakerVerifier.cosineSimilarity(emb1, emb2);
			System.out.printf("[一致性] 同音频两次提取 cos: %.6f%n", score);

			// 同一音频应完全一致（cos = 1.0）
			assertEquals(1.0f, score, 1e-5, "同一音频 embedding 应完全一致");
		}
	}
}
