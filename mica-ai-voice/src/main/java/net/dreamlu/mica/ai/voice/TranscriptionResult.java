package net.dreamlu.mica.ai.voice;

import java.util.List;

/**
 * 完整的转录结果。
 *
 * @param text     最终识别文本
 * @param results  详细的 {@link RecognitionResult} 列表（含时间戳）
 * @param hotwords 识别到的热词列表
 * @param timings  各阶段耗时统计
 */
public record TranscriptionResult(String text,
								  List<RecognitionResult> results,
								  List<String> hotwords,
								  Timings timings) {

	@Override
	public String toString() {
		return String.format("TranscriptionResult{text='%s', hotwords=%s, timings=%.3fs}",
			text, hotwords, timings.total());
	}
}
