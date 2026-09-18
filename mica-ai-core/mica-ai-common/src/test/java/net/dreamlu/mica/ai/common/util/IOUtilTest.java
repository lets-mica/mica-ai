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
package net.dreamlu.mica.ai.common.util;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link IOUtil} 单测。
 */
class IOUtilTest {

	@Test
	void readAllBytesReturnsExactContent() throws IOException {
		byte[] src = new byte[1024];
		for (int i = 0; i < src.length; i++) {
			src[i] = (byte) (i & 0xFF);
		}
		try (InputStream in = new ByteArrayInputStream(src)) {
			byte[] dst = IOUtil.readAllBytes(in);
			assertThat(dst).containsExactly(src);
		}
	}

	@Test
	void readAllBytesOfEmptyStream() throws IOException {
		try (InputStream in = new ByteArrayInputStream(new byte[0])) {
			byte[] dst = IOUtil.readAllBytes(in);
			assertThat(dst).isEmpty();
		}
	}

	@Test
	void readAllBytesPropagatesIOException() {
		InputStream broken = new InputStream() {
			@Override
			public int read() throws IOException {
				throw new IOException("boom");
			}

			@Override
			public int read(byte[] b, int off, int len) throws IOException {
				throw new IOException("boom");
			}
		};
		assertThatThrownBy(() -> IOUtil.readAllBytes(broken))
			.isInstanceOf(IOException.class)
			.hasMessage("boom");
	}
}