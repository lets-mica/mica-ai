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
package net.dreamlu.mica.ai.detect.postprocess;

import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.detect.model.DetectBox;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * 贪心 NMS（与 OpenCV {@code cv::dnn::NMSBoxes} / {@code NMSBoxesBatched} 行为一致）。
 *
 * <p>纯算法实现，不依赖 OpenCV，便于单测覆盖；线程安全（无状态）。
 */
@UtilityClass
public class Nms {

	/**
	 * 按得分降序做贪心 NMS。
	 *
	 * @param boxes          候选框（不可为 null）
	 * @param iouThreshold   IoU 阈值，大于该值的重叠框被抑制
	 * @param classAgnostic  {@code true} 时跨类别互相抑制（NanoDet 参考实现行为）；
	 *                       {@code false} 时仅同类互抑（YOLOX 参考实现行为）
	 * @param maxDetections  最多保留的框数，{@code <=0} 表示不截断
	 * @return 保留下来的框，按得分降序
	 */
	public static List<DetectBox> suppress(List<DetectBox> boxes, float iouThreshold,
										   boolean classAgnostic, int maxDetections) {
		if (boxes == null || boxes.isEmpty()) {
			return Collections.emptyList();
		}
		List<DetectBox> sorted = new ArrayList<>(boxes);
		sorted.sort(Comparator.comparingDouble(DetectBox::getScore).reversed());

		List<DetectBox> kept = new ArrayList<>(sorted.size());
		for (DetectBox candidate : sorted) {
			boolean suppressed = false;
			for (DetectBox chosen : kept) {
				if (!classAgnostic && chosen.getClassId() != candidate.getClassId()) {
					continue;
				}
				if (chosen.iou(candidate) > iouThreshold) {
					suppressed = true;
					break;
				}
			}
			if (suppressed) {
				continue;
			}
			kept.add(candidate);
			if (maxDetections > 0 && kept.size() >= maxDetections) {
				break;
			}
		}
		return kept;
	}
}
