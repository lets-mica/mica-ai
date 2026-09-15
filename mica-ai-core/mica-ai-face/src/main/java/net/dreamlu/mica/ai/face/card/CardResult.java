/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.card;

import lombok.Data;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;

@Data
public class CardResult {

	private Mat image;
	private float[][] quad;
	private double aspectRatio;
	private double score;
	private int rotationDegrees;
	private boolean autoOriented;
	private double cardSize;
	private double sharpness;
	private boolean usable;
	private String unusableReason;
	private int index;

	public void release() {
		ImageUtils.releaseAll(image);
	}
}