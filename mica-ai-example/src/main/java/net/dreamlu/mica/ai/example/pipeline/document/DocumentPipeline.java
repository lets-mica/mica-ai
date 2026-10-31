/*
 * Copyright (c) 2019-2029, Dreamlu 卢春梦 (596392912@qq.com & dreamlu.net).
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.dreamlu.mica.ai.example.pipeline.document;

import lombok.extern.slf4j.Slf4j;
import net.dreamlu.mica.ai.classify.core.ClassifyEngine;
import net.dreamlu.mica.ai.classify.model.ClassifyResult;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.docalign.core.DocAlignEngine;
import net.dreamlu.mica.ai.docalign.model.DocAlignResult;
import net.dreamlu.mica.ai.docalign.model.DocCorner;
import net.dreamlu.mica.ai.docalign.util.DocAlignImageUtils;
import net.dreamlu.mica.ai.enhance.core.EnhanceEngine;
import net.dreamlu.mica.ai.example.config.ClassifyModelRegistry;
import net.dreamlu.mica.ai.layout.model.LayoutLabel;
import net.dreamlu.mica.ai.layout.model.LayoutResult;
import net.dreamlu.mica.ai.layout.pipeline.LayoutPipeline;
import net.dreamlu.mica.ai.textline.core.TextLineEngine;
import net.dreamlu.mica.ai.textline.model.TextLineOrientationResult;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Range;

import java.util.*;

/**
 * 智能文档流水线：把「手机拍的文档」变成「可读、结构正确、方向正确」的图。
 *
 * <p>串联 5 个模型能力，每一步都是独立模型的一次推理：
 * <ol>
 *   <li><b>拉正</b>——{@code mica-ai-docalign} 检四角 + 透视拉平，把斜拍的文档拧正</li>
 *   <li><b>整图方向</b>——{@code mica-ai-classify}（PULC {@code text_image_orientation}）
 *       判 0° / 90° / 180° / 270° 并旋转到位</li>
 *   <li><b>版面分析</b>——{@code mica-ai-layout} 出区域 + 阅读顺序（模型 125MB 不入库，
 *       未启用时本阶段自动降级为空）</li>
 *   <li><b>文本块转正</b>——{@code mica-ai-textline} 对文本块判 0° / 180°，倒置的原地转正</li>
 *   <li><b>画质增强</b>——{@code mica-ai-classify}（PULC {@code clarity_assessment}）判清晰 /
 *       模糊，模糊则 {@code mica-ai-enhance} 超分</li>
 * </ol>
 *
 * <p><b>阶段顺序的理由</b>：几何矫正与语义理解（版面 / 方向）都放在原分辨率上跑，
 * 最快；画质增强放最后，对整图一次性超分，不会让后续检测在 4 倍图上变慢。
 *
 * <p><b>方向标签的约定</b>：PULC {@code text_image_orientation} 的标签是
 * <b>图片当前相对正立的顺时针角度</b>（实测：顺时针转 90° 的图预测为 {@code 90}），
 * 所以修正要转 {@code 360 - label} 度，等价于<b>逆时针转 label 度</b>。
 * 写成「顺时针转 label 度」在 90° / 270° 上正好转反。
 *
 * <p>本类位于 example 模块，作为参考实现；业务模块可按需复制改造。
 *
 * @author DreamLu
 */
@Slf4j
public class DocumentPipeline {

	/** 整图方向模型的标准标签集 */
	private static final Set<String> ORIENTATION_LABELS =
		new HashSet<>(Arrays.asList("0", "90", "180", "270"));

	/** 清晰度模型的标准标签集 */
	private static final Set<String> CLARITY_LABELS =
		new HashSet<>(Arrays.asList("clear", "blur"));

	/**
	 * 参与文本方向判定的版面类别。
	 *
	 * <p>排除 {@code vertical_text}（竖排文本，旋转 180° 无意义）与
	 * {@code table} / {@code image} / {@code seal} 等非文本块。
	 */
	private static final Set<String> TEXT_LABELS = new HashSet<>(Arrays.asList(
		"abstract", "algorithm", "aside_text", "content", "display_formula", "doc_title",
		"footnote", "formula_number", "inline_formula", "number", "paragraph_title",
		"reference", "reference_content", "text"));

	/** 文档拉正引擎 */
	private final DocAlignEngine docAlignEngine;

	/** 整图方向分类引擎（PULC {@code text_image_orientation}） */
	private final ClassifyEngine orientationEngine;

	/** PULC 多模型注册表，用于按需取「清晰度」模型 */
	private final ClassifyModelRegistry classifyRegistry;

	/** 图像超分引擎，可为 null */
	private final EnhanceEngine enhanceEngine;

	/** 版面分析流水线，可为 null（layout 模型未启用时） */
	private final LayoutPipeline layoutPipeline;

	/** 文本行方向引擎，可为 null */
	private final TextLineEngine textLineEngine;

	public DocumentPipeline(DocAlignEngine docAlignEngine,
							ClassifyEngine orientationEngine,
							ClassifyModelRegistry classifyRegistry,
							EnhanceEngine enhanceEngine,
							LayoutPipeline layoutPipeline,
							TextLineEngine textLineEngine) {
		this.docAlignEngine = docAlignEngine;
		this.orientationEngine = orientationEngine;
		this.classifyRegistry = classifyRegistry;
		this.enhanceEngine = enhanceEngine;
		this.layoutPipeline = layoutPipeline;
		this.textLineEngine = textLineEngine;
	}

	/**
	 * 执行整条流水线。
	 *
	 * @param imageBytes 原图字节
	 * @param options    阶段开关
	 * @return 处理结果，含处理后的 PNG 字节与各阶段报告
	 * @throws MicaAiException 解码或推理失败
	 */
	public DocumentResult process(byte[] imageBytes, DocumentOptions options) {
		List<DocumentResult.StageReport> stages = new ArrayList<>();
		// 流水线会不断产生新的 Mat（拉正 / 旋转 / 超分），统一登记后释放，避免提前释放与重复释放
		List<Mat> alive = new ArrayList<>();
		Mat current = null;
		try {
			current = DocAlignImageUtils.byteArrayToMat(imageBytes);
			alive.add(current);

			// ---- 阶段 1：透视拉正 ----
			DocumentResult.AlignReport alignReport = null;
			if (options.isAlign() && docAlignEngine != null) {
				DocAlignResult corners = docAlignEngine.corners(current);
				alignReport = alignReport(corners, current);
				boolean applied = alignReport.getMinCornerScore() >= options.getAlignMinCornerScore();
				if (applied) {
					Mat warped = docAlignEngine.warp(current, corners);
					alive.add(warped);
					current = warped;
					alignReport = alignReport(applied, alignReport.getCorners(),
						alignReport.getMinCornerScore(), current.cols(), current.rows());
					stages.add(stage("align", true, "角点最低分 "
						+ round(alignReport.getMinCornerScore()) + " ≥ "
						+ options.getAlignMinCornerScore() + "，已透视拉平为 "
						+ current.cols() + "x" + current.rows()));
				} else {
					stages.add(stage("align", false, "角点最低分 "
						+ round(alignReport.getMinCornerScore()) + " < "
						+ options.getAlignMinCornerScore() + "，画面中可能没有成形文档，跳过拉正"));
				}
			} else {
				stages.add(stage("align", false, "未启用"));
			}

			// ---- 阶段 2：整图方向 ----
			DocumentResult.OrientationReport orientationReport = null;
			if (options.isOrientation() && orientationEngine != null
				&& hasLabels(orientationEngine, ORIENTATION_LABELS)) {
				ClassifyResult result = orientationEngine.classify(current);
				String label = result.topLabel();
				float score = result.topScore();
				boolean rotated = !"0".equals(label);
				String rotation = null;
				if (rotated) {
					// 标签 = 图片当前的顺时针角度 -> 修正 = 逆时针转 label 度
					rotation = rotateDescription(label);
					Mat turned = new Mat();
					Core.rotate(current, turned, rotateCode(label));
					alive.add(turned);
					current = turned;
				}
				orientationReport = new DocumentResult.OrientationReport(
					label, score, rotated, rotation);
				stages.add(stage("orientation", rotated,
					rotated ? ("预测 " + label + "°（" + round(score) + "），" + rotation)
						: ("预测 " + label + "°（" + round(score) + "），方向已正")));
			} else {
				stages.add(stage("orientation", false,
					options.isOrientation() ? "方向分类模型不可用或标签集不匹配（需 0/90/180/270），跳过"
						: "未启用"));
			}

			// ---- 阶段 3：版面分析 ----
			List<DocumentResult.LayoutBlockReport> blocks = new ArrayList<>();
			if (options.isLayout() && layoutPipeline != null) {
				List<LayoutResult> results = layoutPipeline.detect(current);
				for (LayoutResult item : results) {
					blocks.add(new DocumentResult.LayoutBlockReport(
						item.getLabelCode(),
						item.getScore(),
						item.getBoundingBox(),
						item.getReadingOrder()));
				}
				stages.add(stage("layout", true, "检出 " + blocks.size() + " 个版面区域"));
			} else {
				stages.add(stage("layout", false,
					options.isLayout() ? "layout 模型未启用（125MB 不随仓库分发），跳过版面分析"
						: "未启用"));
			}

			// ---- 阶段 4：文本块倒置转正 ----
			List<DocumentResult.TextBlockReport> textBlocks = new ArrayList<>();
			if (options.isTextLine() && textLineEngine != null && !blocks.isEmpty()) {
				int limit = Math.min(options.getMaxTextBlocks(), blocks.size());
				int flipped = 0;
				for (int i = 0; i < limit; i++) {
					DocumentResult.LayoutBlockReport block = blocks.get(i);
					if (!TEXT_LABELS.contains(block.getLabel())) {
						continue;
					}
					DocumentResult.TextBlockReport report = fixTextBlock(current, block.getBoundingBox());
					if (report != null) {
						textBlocks.add(report);
						if (report.isFlipped()) {
							flipped++;
						}
					}
				}
				stages.add(stage("textLine", true, "判定 " + textBlocks.size()
					+ " 个文本块，其中 " + flipped + " 个倒置并已转正"));
			} else {
				stages.add(stage("textLine", false,
					options.isTextLine() ? (blocks.isEmpty() ? "没有版面结果可供判定，跳过"
						: "文本行方向引擎不可用，跳过") : "未启用"));
			}

			// ---- 阶段 5：清晰度与超分 ----
			DocumentResult.ClarityReport clarityReport = null;
			ClassifyEngine clarityEngine = clarityEngine();
			if (options.isEnhance() && clarityEngine != null
				&& hasLabels(clarityEngine, CLARITY_LABELS)) {
				ClassifyResult result = clarityEngine.classify(current);
				String label = result.topLabel();
				float score = result.topScore();
				// 门限以「clear 概率」表述：clear 概率低于阈值即判为模糊
				float clearScore = result.scoreOf("clear");
				boolean blurry = clearScore < options.getEnhanceBlurThreshold();
				boolean enhanced = false;
				int scale = 1;
				if (blurry && enhanceEngine != null) {
					int before = current.cols();
					Mat upscaled = enhanceEngine.upscale(current);
					alive.add(upscaled);
					current = upscaled;
					enhanced = true;
					scale = before > 0 ? Math.max(1, current.cols() / before) : 1;
				}
				clarityReport = new DocumentResult.ClarityReport(label, score, enhanced, scale);
				stages.add(stage("clarity", enhanced,
					enhanced ? ("判定 " + label + "（" + round(score) + "，clear 概率 "
						+ round(clearScore) + "），已超分 " + scale + " 倍")
						: ("判定 " + label + "（" + round(score) + "，clear 概率 "
						+ round(clearScore) + "），无需超分")));
			} else {
				stages.add(stage("clarity", false,
					options.isEnhance() ? "清晰度模型不可用或标签集不匹配（需 clear/blur），跳过"
						: "未启用"));
			}

			byte[] png = DocAlignImageUtils.imencodePng(current);
			return new DocumentResult(png, current.cols(), current.rows(), stages,
				alignReport, orientationReport, clarityReport, blocks, textBlocks);
		} finally {
			for (Mat mat : alive) {
				if (mat != null) {
					mat.release();
				}
			}
		}
	}

	/**
	 * 对单个文本块判方向，倒置则原地转正。
	 *
	 * @return 判定结果；区域过小或非法时返回 null
	 */
	private DocumentResult.TextBlockReport fixTextBlock(Mat image, int[] bbox) {
		if (bbox == null || bbox.length < 4) {
			return null;
		}
		int x1 = Math.max(0, bbox[0]);
		int y1 = Math.max(0, bbox[1]);
		int x2 = Math.min(image.cols(), bbox[2]);
		int y2 = Math.min(image.rows(), bbox[3]);
		if (x2 - x1 < 8 || y2 - y1 < 8) {
			return null;
		}
		Mat region = image.submat(new Range(y1, y2), new Range(x1, x2));
		Mat rotated = null;
		try {
			TextLineOrientationResult orientation = textLineEngine.classify(region);
			boolean flipped = orientation.getOrientation().isUpsideDown();
			if (flipped) {
				rotated = new Mat();
				Core.rotate(region, rotated, Core.ROTATE_180);
				rotated.copyTo(region);
			}
			return new DocumentResult.TextBlockReport(
				new int[]{x1, y1, x2, y2},
				orientation.getOrientation().getLabel(),
				orientation.getScore(),
				flipped);
		} finally {
			if (rotated != null) {
				rotated.release();
			}
		}
	}

	/**
	 * 取「清晰度」模型引擎；未声明时返回 null（不阻断流水线）。
	 */
	private ClassifyEngine clarityEngine() {
		if (classifyRegistry == null || !classifyRegistry.modelNames().contains("clarity")) {
			return null;
		}
		try {
			return classifyRegistry.get("clarity");
		} catch (MicaAiException e) {
			log.warn("加载 clarity 分类模型失败，跳过清晰度阶段: {}", e.getMessage());
			return null;
		}
	}

	/**
	 * 校验分类引擎的标签集，避免把「整图方向」和「清晰度」模型装反。
	 */
	private static boolean hasLabels(ClassifyEngine engine, Set<String> expected) {
		String[] labels = engine.labels();
		if (labels == null || labels.length != expected.size()) {
			return false;
		}
		Set<String> actual = new LinkedHashSet<>(Arrays.asList(labels));
		return actual.equals(expected);
	}

	/**
	 * 组装拉正阶段报告（角点坐标 + 最低分）。
	 */
	private static DocumentResult.AlignReport alignReport(DocAlignResult corners, Mat image) {
		List<DocumentResult.Corner> list = new ArrayList<>();
		double min = Double.MAX_VALUE;
		if (corners != null) {
			DocCorner lt = corners.getTopLeft();
			DocCorner rt = corners.getTopRight();
			DocCorner rb = corners.getBottomRight();
			DocCorner lb = corners.getBottomLeft();
			list.add(corner(lt, "左上"));
			list.add(corner(rt, "右上"));
			list.add(corner(rb, "右下"));
			list.add(corner(lb, "左下"));
			min = Math.min(Math.min(lt.getScore(), rt.getScore()),
				Math.min(rb.getScore(), lb.getScore()));
		}
		if (min == Double.MAX_VALUE) {
			min = 0d;
		}
		return new DocumentResult.AlignReport(false, list, min,
			image == null ? 0 : image.cols(), image == null ? 0 : image.rows());
	}

	/**
	 * 重新组装拉正报告（拉正后输出尺寸会变，需要回填）。
	 */
	private static DocumentResult.AlignReport alignReport(boolean applied,
														 List<DocumentResult.Corner> corners,
														 double minCornerScore,
														 int width, int height) {
		return new DocumentResult.AlignReport(applied, corners, minCornerScore, width, height);
	}

	private static DocumentResult.Corner corner(DocCorner corner, String position) {
		return new DocumentResult.Corner(corner.getX(), corner.getY(), corner.getScore(), position);
	}

	/**
	 * 标签（图片当前的顺时针角度）对应的转正动作。
	 */
	private static int rotateCode(String label) {
		if ("90".equals(label)) {
			return Core.ROTATE_90_COUNTERCLOCKWISE;
		}
		if ("270".equals(label)) {
			return Core.ROTATE_90_CLOCKWISE;
		}
		return Core.ROTATE_180;
	}

	private static String rotateDescription(String label) {
		if ("90".equals(label)) {
			return "逆时针旋转 90°";
		}
		if ("270".equals(label)) {
			return "顺时针旋转 90°";
		}
		return "旋转 180°";
	}

	private static DocumentResult.StageReport stage(String name, boolean applied, String detail) {
		return new DocumentResult.StageReport(name, applied, detail);
	}

	private static double round(double value) {
		if (Double.isNaN(value) || Double.isInfinite(value)) {
			return 0d;
		}
		return Math.round(value * 1000d) / 1000d;
	}

	/**
	 * 版面类别 code 转可读名（枚举常量名）。
	 *
	 * @param code 版面类别 code
	 * @return 可读名，未知类别返回原 code
	 */
	public static String layoutLabelName(String code) {
		if (code == null) {
			return null;
		}
		LayoutLabel label = LayoutLabel.of(code);
		return label == null ? code : label.name();
	}

}
