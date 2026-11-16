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

import net.dreamlu.mica.ai.track.model.TrackEvent;

import java.util.List;

/**
 * 逐帧行为规则。
 *
 * <p>规则是**有状态**的（保存上一帧的侧别 / 进入时间 / 上报帧号），
 * 一路视频流必须独占一组规则实例；换流或回溯时调 {@link #reset()}。
 */
public interface TrackEventRule {

	default String name() {
		return getClass().getSimpleName();
	}

	/**
	 * 处理一帧，把产生的事件追加到 {@code sink}。
	 */
	void onFrame(FrameContext context, List<TrackEvent> sink);

	/**
	 * 清空规则内部状态。
	 */
	default void reset() {
	}
}
