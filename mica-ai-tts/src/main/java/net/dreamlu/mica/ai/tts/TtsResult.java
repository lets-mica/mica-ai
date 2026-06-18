package net.dreamlu.mica.ai.tts;

/**
 * TTS 合成结果。
 *
 * @param audio      合成的音频数据（float32，范围约 [-1, 1]）
 * @param sampleRate 采样率（24000 Hz）
 * @param duration   音频时长（秒）
 */
public record TtsResult(float[] audio, int sampleRate, double duration) {
}
