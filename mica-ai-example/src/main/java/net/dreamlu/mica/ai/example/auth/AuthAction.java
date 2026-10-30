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
package net.dreamlu.mica.ai.example.auth;

/**
 * 认证挑战动作。
 *
 * <p>YuNet 只输出 5 个关键点（双眼 / 鼻尖 / 嘴角），无法判断眨眼与张口，
 * 因此本示例只用「头部运动幅度」做动作挑战：转头看 yaw 变化范围，点头看 pitch 变化范围。
 *
 * <p>生产环境应扩充动作库（红外 / 3D 结构光 / 眨眼）并随机抽取多条动作。
 */
public enum AuthAction {

	/**
	 * 左右转头：鼻尖相对双眼中点的水平偏移（以眼距归一化）的极差。
	 */
	TURN_HEAD("turn", "请左右转头", 0.20f),

	/**
	 * 上下点头：鼻尖相对双眼中点的垂直偏移（以眼距归一化）的极差。
	 */
	NOD_HEAD("nod", "请上下点头", 0.12f);

	private final String code;
	private final String label;
	private final float requiredRange;

	AuthAction(String code, String label, float requiredRange) {
		this.code = code;
		this.label = label;
		this.requiredRange = requiredRange;
	}

	public String getCode() {
		return code;
	}

	public String getLabel() {
		return label;
	}

	public float getRequiredRange() {
		return requiredRange;
	}
}
