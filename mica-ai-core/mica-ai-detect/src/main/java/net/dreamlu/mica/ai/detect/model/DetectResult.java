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
package net.dreamlu.mica.ai.detect.model;

import lombok.Getter;

import java.util.Collections;
import java.util.List;

/**
 * 单帧检测结果（不可变快照，无需 release）。
 */
@Getter
public class DetectResult {

	private final int width;
	private final int height;
	private final List<DetectBox> boxes;

	public DetectResult(int width, int height, List<DetectBox> boxes) {
		this.width = width;
		this.height = height;
		this.boxes = boxes == null ? Collections.<DetectBox>emptyList()
			: Collections.unmodifiableList(boxes);
	}

	/**
	 * 检测框数量。
	 *
	 * @return 框数
	 */
	public int size() {
		return boxes.size();
	}
}
