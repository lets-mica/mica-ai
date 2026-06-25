/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.engine;

import net.dreamlu.mica.ai.face.config.FaceBox;

import java.awt.image.BufferedImage;
import java.util.List;

/**
 * 人脸检测器抽象。
 *
 * <p>实现该接口即可接入 mica-ai-face 的人脸识别主流程，
 * 内置默认实现 {@link YuNetDetector}（OpenCV Zoo，Apache-2.0，可商用）。
 *
 * <p>典型应用：自定义训练的人脸检测器、自家业务的 ROI 检测逻辑等。
 *
 * @author dreamlu
 * @since 1.0.0
 */
public interface FaceDetector extends AutoCloseable {

	/**
	 * 检测图片中的人脸。
	 *
	 * <p>坐标系：原图坐标系（左上角为原点，y 向下）。
	 *
	 * @param image 待检测图片
	 * @return 人脸框列表，按 score 降序；含 5 个关键点（眼睛、鼻尖、嘴角）
	 */
	List<FaceBox> detect(BufferedImage image);

	@Override
	void close();
}