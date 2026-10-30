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

import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 内存版认证会话存储。
 *
 * <p>仅供示例 / 单机 Demo：多实例部署请换成 Redis（带 TTL 与一次性原子删除）。
 * 到达容量上限时先清理过期会话，仍超限则淘汰最老的一条。
 */
@Component
public class FaceAuthStore {

	private static final int MAX_SESSIONS = 1024;

	private final Map<String, FaceAuthSession> sessions = new ConcurrentHashMap<>();

	public FaceAuthSession create(List<AuthAction> actions, long ttlMs, int maxFrames, long frameIntervalMs) {
		purge();
		long now = System.currentTimeMillis();
		String id = UUID.randomUUID().toString().replace("-", "");
		FaceAuthSession session = new FaceAuthSession(id, now, now + ttlMs, actions, frameIntervalMs, maxFrames);
		sessions.put(id, session);
		return session;
	}

	/**
	 * 读取会话，过期即删除并返回 null。
	 */
	public FaceAuthSession get(String sessionId) {
		if (sessionId == null || sessionId.isEmpty()) {
			return null;
		}
		FaceAuthSession session = sessions.get(sessionId);
		if (session == null) {
			return null;
		}
		if (session.isExpired(System.currentTimeMillis())) {
			sessions.remove(sessionId);
			return null;
		}
		return session;
	}

	/**
	 * 一次性消费会话（submit 用），返回 null 表示不存在或已过期。
	 */
	public FaceAuthSession consume(String sessionId) {
		FaceAuthSession session = get(sessionId);
		if (session == null) {
			return null;
		}
		return sessions.remove(sessionId);
	}

	private void purge() {
		if (sessions.size() < MAX_SESSIONS) {
			return;
		}
		long now = System.currentTimeMillis();
		sessions.values().removeIf(s -> s.isExpired(now));
		if (sessions.size() < MAX_SESSIONS) {
			return;
		}
		String oldestId = null;
		long oldest = Long.MAX_VALUE;
		for (Map.Entry<String, FaceAuthSession> entry : sessions.entrySet()) {
			if (entry.getValue().getCreatedAt() < oldest) {
				oldest = entry.getValue().getCreatedAt();
				oldestId = entry.getKey();
			}
		}
		if (oldestId != null) {
			sessions.remove(oldestId);
		}
	}
}
