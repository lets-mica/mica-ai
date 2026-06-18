package net.dreamlu.mica.ai.voice;

/**
 * 单个识别单元结果（字符或 Token 块）。
 *
 * @param text     识别文本（字符或块）
 * @param start    起始时间（秒）
 * @param hotword  是否为命中的热词
 */
public record RecognitionResult(String text, double start, boolean hotword) {

	@Override
	public String toString() {
		return String.format("RecognitionResult{text='%s', start=%.3f, hotword=%s}", text, start, hotword);
	}
}
