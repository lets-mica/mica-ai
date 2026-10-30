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
package net.dreamlu.mica.ai.classify.util;

import net.dreamlu.mica.ai.common.exception.MicaAiException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * {@link ClassifyLabelUtils} 单元测试。
 */
class ClassifyLabelUtilsTest {

	@Test
	void readsLinesInOrderAndKeepsDuplicates(@TempDir Path dir) throws IOException {
		Path file = dir.resolve("labels.txt");
		Files.write(file, "clear\nblur\n\n  clear  \n".getBytes(StandardCharsets.UTF_8));

		String[] labels = ClassifyLabelUtils.readLabels(file.toString());

		assertThat(labels).containsExactly("clear", "blur", "clear");
	}

	@Test
	void rejectsNullAndEmptyPath() {
		assertThatThrownBy(() -> ClassifyLabelUtils.readLabels(null))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("labelsPath");
		assertThatThrownBy(() -> ClassifyLabelUtils.readLabels(""))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("labelsPath");
	}

	@Test
	void rejectsMissingFile(@TempDir Path dir) {
		Path missing = dir.resolve("missing.txt");

		assertThatThrownBy(() -> ClassifyLabelUtils.readLabels(missing.toString()))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("不存在");
	}

	@Test
	void rejectsBlankOnlyFile(@TempDir Path dir) throws IOException {
		Path file = dir.resolve("blank.txt");
		Files.write(file, "\n   \n\n".getBytes(StandardCharsets.UTF_8));

		assertThatThrownBy(() -> ClassifyLabelUtils.readLabels(file.toString()))
			.isInstanceOf(MicaAiException.class)
			.hasMessageContaining("为空");
	}
}
