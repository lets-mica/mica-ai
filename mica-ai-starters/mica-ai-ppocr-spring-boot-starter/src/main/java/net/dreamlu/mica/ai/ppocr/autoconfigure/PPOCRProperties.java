package net.dreamlu.mica.ai.ppocr.autoconfigure;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * PP-OCR 配置属性。
 *
 * <p>对应 {@code mica.ai.ppocr} 配置前缀。
 */
@Data
@ConfigurationProperties(prefix = "mica.ai.ppocr")
public class PPOCRProperties {

	/**
	 * 是否启用该 Starter。默认 {@code true}：启用时必填的 det/rec 模型路径及字典缺失将启动失败；
	 * 设为 {@code false} 时整个 Starter 不注入任何 Bean。
	 */
	private boolean enabled = true;

	/** 检测模型路径（必填） */
	private String detModelPath;

	/** 识别模型路径（必填） */
	private String recModelPath;

	/** 识别字符字典路径（必填） */
	private String recCharDictPath;

	/** 检测图像短边限制 */
	private int detLimitSideLen = 64;

	/** 检测限制类型: min / max */
	private String detLimitType = "min";

	/** 检测最大边长限制 */
	private int detMaxSideLimit = 4000;

	/** 检测阈值 */
	private float detThresh = 0.3f;

	/** 检测框阈值 */
	private float detBoxThresh = 0.6f;

	/** 检测 unclip 比例 */
	private float detUnclipRatio = 1.5f;

	/** 识别输入 shape [C, H, W] */
	private int[] recImageShape = {3, 48, 320};

	/** 识别批处理大小 */
	private int recBatchSize = 6;

	/** 是否优先 CPU（跨平台 bit-exact） */
	private boolean preferAccelerator = false;

	/** ONNX 内部线程数 */
	private int intraOpNumThreads = 1;

	/** ONNX 交互线程数 */
	private int interOpNumThreads = 1;
}
