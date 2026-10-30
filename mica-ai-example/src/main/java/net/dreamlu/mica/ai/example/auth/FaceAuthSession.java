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

import lombok.Getter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 单次人脸认证会话。
 *
 * <p>只保存状态与最佳帧特征，**不保存任何原始图像**，满足「原图不落库」的最小要求。
 * 会话一次性消费，submit 后立即从 {@link FaceAuthStore} 移除。
 *
 * <p>所有可变状态通过 {@code synchronized} 串行化，允许端上并发重试上传同一会话的帧。
 */
@Getter
public class FaceAuthSession {

	private final String sessionId;
	private final long createdAt;
	private final long expiresAt;
	private final List<AuthAction> actions;
	private final long frameIntervalMs;
	private final int maxFrames;

	private int frameCount;
	private float yawMin = Float.MAX_VALUE;
	private float yawMax = -Float.MAX_VALUE;
	private float pitchMin = Float.MAX_VALUE;
	private float pitchMax = -Float.MAX_VALUE;
	private float[] bestFeature;
	private float bestLiveScore = -1f;

	public FaceAuthSession(String sessionId, long createdAt, long expiresAt,
						   List<AuthAction> actions, long frameIntervalMs, int maxFrames) {
		this.sessionId = sessionId;
		this.createdAt = createdAt;
		this.expiresAt = expiresAt;
		this.actions = Collections.unmodifiableList(new ArrayList<>(actions));
		this.frameIntervalMs = frameIntervalMs;
		this.maxFrames = maxFrames;
	}

	public boolean isExpired(long now) {
		return now > expiresAt;
	}

	/**
	 * 帧计数 +1，返回当前帧序号（从 1 开始）。
	 */
	public synchronized int beginFrame() {
		frameCount++;
		return frameCount;
	}

	/**
	 * 记录一帧的姿态比例（yaw / pitch 均已按眼距归一化）。
	 */
	public synchronized void observePose(float yaw, float pitch) {
		if (yaw < yawMin) {
			yawMin = yaw;
		}
		if (yaw > yawMax) {
			yawMax = yaw;
		}
		if (pitch < pitchMin) {
			pitchMin = pitch;
		}
		if (pitch > pitchMax) {
			pitchMax = pitch;
		}
	}

	/**
	 * 保留活体分最高的那帧特征，用于 submit 时做 1:1。
	 */
	public synchronized void offerFeature(float[] feature, float liveScore) {
		if (feature == null) {
			return;
		}
		if (liveScore > bestLiveScore) {
			bestLiveScore = liveScore;
			bestFeature = feature.clone();
		}
	}

	public synchronized float[] bestFeature() {
		return bestFeature;
	}

	public synchronized float progress(AuthAction action) {
		float range;
		if (action == AuthAction.TURN_HEAD) {
			range = yawMax - yawMin;
		} else {
			range = pitchMax - pitchMin;
		}
		if (range <= 0f || Float.isNaN(range)) {
			return 0f;
		}
		float p = range / action.getRequiredRange();
		return p > 1f ? 1f : p;
	}

	public synchronized boolean satisfied(AuthAction action) {
		return progress(action) >= 1f;
	}

	public synchronized boolean isComplete() {
		for (AuthAction action : actions) {
			if (!satisfied(action)) {
				return false;
			}
		}
		return frameCount >= 2;
	}
}
