/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.common.util;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * IO 工具（跨能力通用基础设施）。
 *
 * @author L.cm
 */
public final class IOUtil {

	private IOUtil() {
	}

	public static byte[] readAllBytes(InputStream in) throws IOException {
		ByteArrayOutputStream out = new ByteArrayOutputStream(Math.max(64, in.available()));
		byte[] buf = new byte[8192];
		int n;
		while ((n = in.read(buf)) != -1) {
			out.write(buf, 0, n);
		}
		return out.toByteArray();
	}
}
