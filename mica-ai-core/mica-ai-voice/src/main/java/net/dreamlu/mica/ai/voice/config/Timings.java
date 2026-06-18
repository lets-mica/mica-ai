package net.dreamlu.mica.ai.voice.config;

/**
 * 各阶段耗时统计（单位：秒）。
 *
 * @param frontend  特征提取耗时
 * @param encoder   编码器推理耗时
 * @param decoder   解码器 (CTC) 推理耗时
 * @param radar     热词雷达扫描耗时
 * @param integrate 结果整合耗时
 * @param total     总耗时
 */
public record Timings(double frontend, double encoder, double decoder,
					  double radar, double integrate, double total) {

	/**
	 * 全零耗时（用于分段合并后的占位）。
	 */
	public static Timings empty() {
		return new Timings(0, 0, 0, 0, 0, 0);
	}
}
