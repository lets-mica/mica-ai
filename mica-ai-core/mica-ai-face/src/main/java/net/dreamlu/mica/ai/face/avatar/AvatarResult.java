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