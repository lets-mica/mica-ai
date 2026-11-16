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
package net.dreamlu.mica.ai.track.event;

import net.dreamlu.mica.ai.track.model.BBox;

/**
 * ROI 判定使用的目标锚点。
 *
 * <p>行人 / 车辆做区域判定时，用底部中心（脚点 / 轮胎接地点）通常比几何中心稳定，
 * 因为框的上半部分容易随姿态或遮挡变化。
 */
public enum AnchorPolicy {

	CENTER {
		@Override
		public float anchorX(BBox box) {
			return box.cx();
		}

		@Override
		public float anchorY(BBox box) {
			return box.cy();
		}
	},

	BOTTOM_CENTER {
		@Override
		public float anchorX(BBox box) {
			return box.cx();
		}

		@Override
		public float anchorY(BBox box) {
			return box.getY2();
		}
	};

	public abstract float anchorX(BBox box);

	public abstract float anchorY(BBox box);
}
