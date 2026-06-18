package net.dreamlu.mica.ai.voice;

import java.util.List;

/**
 * 完整的转录结果。
 *
 * @param text     最终识别文本
 * @param results  详细的 {@link RecognitionResult} 列表（含时间戳）
 * @param hotWords 识别到的热词列表
 * @param timings  各阶段耗时统计
 */
public record TranscriptionResult(String text,
								  List<RecognitionResult> results,
								  List<String> hotWords,
								  Timings timings) {
}
