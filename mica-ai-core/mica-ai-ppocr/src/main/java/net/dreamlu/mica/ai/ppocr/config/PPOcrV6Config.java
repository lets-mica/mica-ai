package net.dreamlu.mica.ai.ppocr.config;

import lombok.Builder;
import lombok.Getter;

/**
 * PP-OCRv6 引擎配置。
 *
 * <p>使用 Builder 模式构建，所有参数均有合理默认值。
 */
@Getter
@Builder
public final class PPOcrV6Config {

	/** 检测模型路径（必填） */
	private String detModelPath;

	/** 识别模型路径（必填） */
	private String recModelPath;

	/** 识别字符字典路径（必填） */
	private String recCharDictPath;

	/** 检测：短边限制（默认 64） */
	@Builder.Default
	private int detLimitSideLen = 64;

	/** 检测：限制类型，min 或 max */
	@Builder.Default
	private String detLimitType = "min";

	/** 检测：最大边长限制 */
	@Builder.Default
	private int detMaxSideLimit = 4000;

	/** 检测：DB 后处理二值化阈值 */
	@Builder.Default
	private float detThresh = 0.3f;

	/** 检测：DB 后处理 box 阈值 */
	@Builder.Default
	private float detBoxThresh = 0.6f;

	/** 检测：DB 后处理 unclip 比率 */
	@Builder.Default
	private float detUnclipRatio = 1.5f;

	/** 识别：输入图像形状 [C, H, W] */
	@Builder.Default
	private int[] recImageShape = {3, 48, 320};

	/** 识别：批处理大小 */
	@Builder.Default
	private int recBatchSize = 6;

	/** 是否优先使用 GPU 加速（默认 false，强制 CPU） */
	@Builder.Default
	private boolean preferAccelerator = false;

	/** ONNX Runtime 线程数 */
	@Builder.Default
	private int intraOpNumThreads = 1;

	/** ONNX Runtime 线程数 */
	@Builder.Default
	private int interOpNumThreads = 1;

	public static PPOcrV6Config defaults() {
		return builder().build();
	}
}
