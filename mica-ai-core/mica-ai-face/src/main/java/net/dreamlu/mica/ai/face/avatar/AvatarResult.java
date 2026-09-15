/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.avatar;

import lombok.Data;
import net.dreamlu.mica.ai.face.model.FaceBox;
import net.dreamlu.mica.ai.face.util.ImageUtils;
import org.opencv.core.Mat;

@Data
public class AvatarResult {

	private Mat image;
	private FaceBox box;
	private float[][] windowQuad;
	private int size;
	private double faceSize;
	private int orientationDegrees;
	private boolean tiledDetection;
	private double sharpness;
	private boolean usable;
	private String unusableReason;
	private int index;

	public void release() {
		ImageUtils.releaseAll(image);
	}
}