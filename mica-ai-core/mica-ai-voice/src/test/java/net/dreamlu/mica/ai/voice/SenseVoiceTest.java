package net.dreamlu.mica.ai.voice;

import org.junit.jupiter.api.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * SenseVoice 引擎测试 — 参照 Python 16-Final-Hotword-Inference.py。
 *
 * <p>使用 SenseVoice-ONNX 项目导出的 fp32 模型和 test-fun.wav 音频。
 */
class SenseVoiceTest {

	// ====== 路径配置（指向 SenseVoice-ONNX 项目目录） ======
	private static final String SV_ONNX_DIR = "E:\\codes\\ai\\SenseVoice-ONNX";
	private static final String MODEL_DIR = SV_ONNX_DIR + "\\model";
	private static final String ENCODER_PATH = MODEL_DIR + "\\SenseVoice-Encoder.fp32.onnx";
	private static final String DECODER_PATH = MODEL_DIR + "\\SenseVoice-CTC.fp32.onnx";
	private static final String TOKENIZER_PATH = MODEL_DIR + "\\Tokenizer.bpe.model";
	private static final String AUDIO_PATH = SV_ONNX_DIR + "\\test-fun.wav";
	private static final String HOTWORDS_PATH = SV_ONNX_DIR + "\\hot.txt";

	@Test
	void testConfigBuilder() {
		SenseVoiceConfig config = new SenseVoiceConfig()
			.encoderPath("encoder.onnx")
			.decoderPath("decoder.onnx")
			.tokenizerPath("tokenizer.model")
			.hotwords(List.of("mica", "梦想卢"))
			.topK(10)
			.itn(true);

		assertEquals("encoder.onnx", config.getEncoderPath());
		assertEquals("decoder.onnx", config.getDecoderPath());
		assertEquals("tokenizer.model", config.getTokenizerPath());
		assertEquals(List.of("mica", "梦想卢"), config.getHotwords());
		assertEquals(10, config.getTopK());
		assertTrue(config.isItn());
	}

	@Test
	void testConfigDefaults() {
		SenseVoiceConfig config = new SenseVoiceConfig();
		assertEquals("cpu", config.getOnnxProvider());
		assertEquals(10, config.getTopK());
		assertTrue(config.isItn());
		assertNull(config.getHotwords());
	}

	/**
	 * 完整推理测试：加载模型 → 读取热词 → 3 轮测速 → 打印详细结果。
	 * 对应 Python 16-Final-Hotword-Inference.py
	 */
	@Test
	void testRecognizeWithHotwords() throws IOException {
		// 跳过条件：模型文件不存在则跳过
		if (!Files.isRegularFile(Path.of(ENCODER_PATH))) {
			System.out.println("⚠️ 模型文件不存在，跳过推理测试: " + ENCODER_PATH);
			return;
		}
		if (!Files.isRegularFile(Path.of(AUDIO_PATH))) {
			System.out.println("⚠️ 音频文件不存在，跳过推理测试: " + AUDIO_PATH);
			return;
		}

		// 1. 读取热词（同 Python 版逻辑：忽略空行和 # 注释）
		List<String> hotwords = readHotwords(HOTWORDS_PATH);
		System.out.println("[Hotwords] 当前热词列表: " + hotwords.size() + " 个");

		// 2. 初始化引擎
		SenseVoiceConfig config = new SenseVoiceConfig()
			.encoderPath(ENCODER_PATH)
			.decoderPath(DECODER_PATH)
			.tokenizerPath(TOKENIZER_PATH)
			.onnxProvider("cpu")
			.topK(5)
			.hotwords(hotwords);

		try (SenseVoice voice = new SenseVoice(config)) {
			// 3. 加载音频
			float[] audio = SenseVoice.loadWav(AUDIO_PATH);
			System.out.println("[Audio] 采样点数: " + audio.length +
				", 时长: " + String.format("%.2f", audio.length / 16000.0) + "s");

			// 4. 运行 3 轮推理测速
			System.out.println("\n[Performance] 开始测速...");
			TranscriptionResult result = null;
			for (int i = 1; i <= 3; i++) {
				result = voice.recognize(audio);
				Timings tm = result.timings();

				System.out.printf(" >>> 第 %d 轮耗时: %7.2fms | 识别文本: %s%n",
					i, tm.total() * 1000, result.text());
				System.out.printf("     [细节] Frontend: %4.1fms | Encoder: %4.1fms | Decoder: %4.1fms | Radar: %4.1fms%n",
					tm.frontend() * 1000, tm.encoder() * 1000, tm.decoder() * 1000, tm.radar() * 1000);
			}

			// 5. 展示最后一轮的详细结果
			assertNotNull(result);
			System.out.println("\n" + "=".repeat(50));
			System.out.printf("%10s | %-10s | %s%n", "时间戳", "字符", "类型");
			System.out.println("-".repeat(50));
			for (RecognitionResult r : result.results()) {
				String charType = r.hotWord() ? "🔥 HOTWORD" : "  Greedy";
				System.out.printf("  %5.2fs | %-10s | %s%n", r.start(), r.text(), charType);
			}
			System.out.println("-".repeat(50));
			System.out.println("【检测到的热词】: " + result.hotWords());
			System.out.println("【最终识别文本】");
			System.out.println(result.text());
			System.out.println("=".repeat(50));

			// 6. 基本断言
			assertNotNull(result.text());
			assertFalse(result.text().isEmpty(), "识别文本不应为空");
			assertFalse(result.results().isEmpty(), "识别结果列表不应为空");
		}
	}

	/**
	 * 读取热词文件（同 Python 版逻辑）。
	 */
	private List<String> readHotwords(String path) throws IOException {
		if (!Files.isRegularFile(Path.of(path))) {
			System.out.println("⚠️ 热词文件不存在: " + path + ", 使用默认热词");
			return List.of("Fun-ASR-Nano");
		}
		try (BufferedReader reader = Files.newBufferedReader(Path.of(path), StandardCharsets.UTF_8)) {
			return reader.lines()
				.map(String::trim)
				.filter(line -> !line.isEmpty() && !line.startsWith("#"))
				.toList();
		}
	}
}
