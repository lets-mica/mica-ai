/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.filetype;

import net.dreamlu.mica.ai.filetype.model.ContentTypeLabel;
import net.dreamlu.mica.ai.filetype.model.FiletypeResult;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.assertj.core.api.Assertions.assertThat;

@EnabledIf("isModelPresent")
class FiletypeDetectorTest {

	private static final Path MODEL_PATH = repoPath("model-tools/filetype/models/model.onnx");
	private static final Path CONFIG_PATH = repoPath("model-tools/filetype/models/config.min.json");
	private static final Path KB_PATH = repoPath("model-tools/filetype/models/content_types_kb.min.json");

	private static Path repoPath(String relativePath) {
		Path dir = Paths.get("").toAbsolutePath();
		while (dir != null && !Files.exists(dir.resolve("model-tools"))) {
			dir = dir.getParent();
		}
		return dir == null ? Paths.get(relativePath) : dir.resolve(relativePath);
	}

	static boolean isModelPresent() {
		return Files.exists(MODEL_PATH)
			&& Files.exists(CONFIG_PATH)
			&& Files.exists(KB_PATH);
	}

	private FiletypeDetector newDetector() {
		return newDetector(PredictionMode.BEST_GUESS);
	}

	private FiletypeDetector newDetector(PredictionMode mode) {
		FiletypeConfig cfg = FiletypeConfig.builder()
			.modelPath(MODEL_PATH.toString())
			.configPath(CONFIG_PATH.toString())
			.contentTypesPath(KB_PATH.toString())
			.predictionMode(mode)
			.build();
		return new FiletypeDetector(cfg);
	}

	@Test
	void detectBytes_pngHeader() {
		byte[] png = new byte[]{
			(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
			0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
			0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,
			(byte) 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, (byte) 0xF3, (byte) 0xFF
		};
		try (FiletypeDetector detector = newDetector()) {
			FiletypeResult result = detector.detectBytes(png);
			assertThat(result.getModelLabel()).isEqualTo("png");
			assertThat(result.getOutputLabel()).isEqualTo("png");
			assertThat(result.getScore()).isGreaterThan(0.5f);
			assertThat(result.isText()).isFalse();
		}
	}

	@Test
	void detectBytes_jsonContent() {
		StringBuilder sb = new StringBuilder("{\"name\":\"mica-ai-filetype\",\"version\":\"1.0.0\",\"data\":[");
		for (int i = 0; i < 64; i++) {
			sb.append(i).append(",");
		}
		sb.append("]}");
		byte[] json = sb.toString().getBytes(StandardCharsets.UTF_8);
		try (FiletypeDetector detector = newDetector()) {
			FiletypeResult result = detector.detectBytes(json);
			assertThat(result.getModelLabel()).isIn("json", "txt");
		}
	}

	@Test
	void detectBytes_plainText() {
		String chunk = "This is a long enough plain text to give the detector enough "
			+ "features to confidently call it a text file. We are essentially writing "
			+ "lorem ipsum style content so that the feature extractor gets a reasonable "
			+ "block of bytes to look at.";
		byte[] text = repeat(chunk, 4).getBytes(StandardCharsets.UTF_8);
		try (FiletypeDetector detector = newDetector()) {
			FiletypeResult result = detector.detectBytes(text);
			assertThat(result.isText()).isTrue();
			assertThat(result.getModelLabel()).isIn("txt", "markdown", "english", "csv", "json", "xml", "yaml");
		}
	}

	@Test
	void detectBytes_empty() {
		try (FiletypeDetector detector = newDetector()) {
			FiletypeResult result = detector.detectBytes(new byte[0]);
			assertThat(result.getOutputLabel()).isEqualTo(ContentTypeLabel.EMPTY);
			assertThat(result.getScore()).isEqualTo(1.0f);
		}
	}

	@Test
	void detectBytes_fewBytesUtf8_returnsTxt() {
		byte[] shortUtf8 = "abc".getBytes(StandardCharsets.UTF_8);
		try (FiletypeDetector detector = newDetector()) {
			FiletypeResult result = detector.detectBytes(shortUtf8);
			assertThat(result.getOutputLabel()).isEqualTo(ContentTypeLabel.TXT);
			assertThat(result.isText()).isTrue();
		}
	}

	@Test
	void detectBytes_fewBytesBinary_returnsUnknown() {
		byte[] shortBin = new byte[]{(byte) 0xFF, (byte) 0xFE, 0x00};
		try (FiletypeDetector detector = newDetector()) {
			FiletypeResult result = detector.detectBytes(shortBin);
			assertThat(result.getOutputLabel()).isEqualTo(ContentTypeLabel.UNKNOWN);
			assertThat(result.isText()).isFalse();
		}
	}

	@Test
	void detectPath_regularFile(@TempDir Path tempDir) throws IOException {
		Path file = tempDir.resolve("hello.txt");
		String content = repeat("Hello, mica-ai-filetype! This is a sample text file. ", 8);
		Files.write(file, content.getBytes(StandardCharsets.UTF_8));

		try (FiletypeDetector detector = newDetector()) {
			FiletypeResult result = detector.detectPath(file);
			assertThat(result.isText()).isTrue();
		}
	}

	@Test
	void detectPath_directory() throws IOException {
		try (FiletypeDetector detector = newDetector()) {
			FiletypeResult result = detector.detectPath(Paths.get(System.getProperty("java.io.tmpdir")));
			assertThat(result.getOutputLabel()).isEqualTo(ContentTypeLabel.DIRECTORY);
		}
	}

	@Test
	void highConfidenceMode_fallsBackToTxtWhenBelowThreshold() {
		byte[] json = ("{\"items\":[" + repeat("1,2,3,4,5,6,7,8,9,0,", 8) + "]}").getBytes(StandardCharsets.UTF_8);
		try (FiletypeDetector detector = newDetector(PredictionMode.HIGH_CONFIDENCE)) {
			FiletypeResult result = detector.detectBytes(json);
			assertThat(result.getModelLabel()).isIn("json", "txt");
		}
	}

	@Test
	void bestGuessMode_neverFallsBack() {
		byte[] png = new byte[]{
			(byte) 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
			0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
			0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x10,
			(byte) 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, (byte) 0xF3, (byte) 0xFF
		};
		try (FiletypeDetector detector = newDetector(PredictionMode.BEST_GUESS)) {
			FiletypeResult result = detector.detectBytes(png);
			assertThat(result.getOutputLabel()).isEqualTo("png");
		}
	}

	private static String repeat(String s, int count) {
		StringBuilder sb = new StringBuilder(s.length() * count);
		for (int i = 0; i < count; i++) {
			sb.append(s);
		}
		return sb.toString();
	}
}