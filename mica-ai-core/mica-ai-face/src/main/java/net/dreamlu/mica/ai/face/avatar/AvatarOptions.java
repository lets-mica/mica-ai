/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.face.avatar;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.opencv.core.Scalar;

@Data
@Builder(toBuilder = true)
@NoArgsConstructor
@AllArgsConstructor
public class AvatarOptions {

	@Builder.Default
	private int size = 256;

	@Builder.Default
	private double faceScale = 1.6;

	@Builder.Default
	private double verticalOffset = 0.0;

	@Builder.Default
	private boolean deRotate = true;

	@Builder.Default
	private double rotationDegrees = 0.0;

	@Builder.Default
	private boolean autoOrient = true;

	@Builder.Default
	private String background = "#FFFFFF";

	@Builder.Default
	private boolean tileDetect = true;

	@Builder.Default
	private int tileSize = 480;

	@Builder.Default
	private double tileOverlap = 0.30;

	@Builder.Default
	private double tileThreshold = 0.60;

	@Builder.Default
	private int minFaceSize = 40;

	@Builder.Default
	private int maxFaces = 0;

	public static AvatarOptions defaults() {
		return AvatarOptions.builder().build();
	}

	public static AvatarOptions facePhoto() {
		return AvatarOptions.builder().faceScale(1.3).build();
	}

	public static AvatarOptions portrait() {
		return AvatarOptions.builder().faceScale(2.0).build();
	}

	public Scalar backgroundScalar() {
		return parseColor(background);
	}

	static Scalar parseColor(String color) {
		String hex = color == null ? "" : color.trim();
		if (hex.startsWith("#")) {
			hex = hex.substring(1);
		}
		if (hex.length() == 3) {
			StringBuilder sb = new StringBuilder(6);
			for (int i = 0; i < 3; i++) {
				sb.append(hex.charAt(i)).append(hex.charAt(i));
			}
			hex = sb.toString();
		}
		if (hex.length() != 6 || !hex.matches("[0-9a-fA-F]{6}")) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED,
				"background 需为 #RRGGBB 或 #RGB 十六进制色值，实际: " + color);
		}
		int r = Integer.parseInt(hex.substring(0, 2), 16);
		int g = Integer.parseInt(hex.substring(2, 4), 16);
		int b = Integer.parseInt(hex.substring(4, 6), 16);
		return new Scalar(b, g, r);
	}

	void validate() {
		if (size < 16 || size > 4096) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED, "size 需在 [16, 4096]，实际: " + size);
		}
		if (!(faceScale >= 1.0 && faceScale <= 8.0)) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED, "faceScale 需在 [1.0, 8.0]，实际: " + faceScale);
		}
		if (!(verticalOffset >= -1.0 && verticalOffset <= 1.0)) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED,
				"verticalOffset 需在 [-1.0, 1.0]，实际: " + verticalOffset);
		}
		if (!(rotationDegrees >= -180.0 && rotationDegrees <= 180.0)) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED,
				"rotationDegrees 需在 [-180, 180]，实际: " + rotationDegrees);
		}
		if (tileSize < 32 || tileSize > 8192) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED, "tileSize 需在 [32, 8192]，实际: " + tileSize);
		}
		if (!(tileOverlap >= 0.0 && tileOverlap < 1.0)) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED, "tileOverlap 需在 [0, 1)，实际: " + tileOverlap);
		}
		if (!(tileThreshold > 0.0 && tileThreshold <= 1.0)) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED,
				"tileThreshold 需在 (0, 1]，实际: " + tileThreshold);
		}
		if (minFaceSize < 0) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED, "minFaceSize 不能为负");
		}
		if (maxFaces < 0) {
			throw new MicaAiException(
				MicaAiException.ErrorCode.AVATAR_FAILED, "maxFaces 不能为负");
		}
		backgroundScalar();
	}
}