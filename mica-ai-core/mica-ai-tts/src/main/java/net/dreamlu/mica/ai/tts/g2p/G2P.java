package net.dreamlu.mica.ai.tts.g2p;

import net.dreamlu.mica.ai.tts.KokoroTtsConfig;

/**
 * G2P（Grapheme-to-Phoneme）转换器接口。
 * <p>负责将输入文本转换为 Kokoro 模型可识别的音素序列。
 * <p>音素格式：bopomofo（注音符号）用于中文，IPA 用于英文。
 *
 * <p>用户可实现自己的 G2P 并通过 {@link KokoroTtsConfig.Builder#g2p(G2P)} 注入。
 *
 * @author L.cm
 */
@FunctionalInterface
public interface G2P {

	/**
	 * 将输入文本转换为音素字符串。
	 *
	 * @param text 输入文本（中/英/数字混合）
	 * @return Kokoro 可识别的音素序列
	 */
	String convert(String text);

}
