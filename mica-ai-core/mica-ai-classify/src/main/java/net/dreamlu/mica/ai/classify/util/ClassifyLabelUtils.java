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

import lombok.experimental.UtilityClass;
import net.dreamlu.mica.ai.common.exception.ErrorCode;
import net.dreamlu.mica.ai.common.exception.MicaAiException;
import net.dreamlu.mica.ai.common.onnx.OnnxModelSession;
import net.dreamlu.mica.ai.common.util.IOUtil;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * label_list 读取工具（零 Spring、零 OpenCV）。
 *
 * <p>格式约定：一行一个标签，按行序对应模型输出索引；空行忽略，
 * <b>重复标签保留</b>（去重会让索引整体错位，属于静默错误）。
 * 工具类，私有构造，线程安全（无状态）。
 */
@UtilityClass
public class ClassifyLabelUtils {

	/**
	 * 读取 label_list 文件。
	 *
	 * @param labelsPath 文件路径（必填）；支持 {@code classpath:} 前缀，其余按文件系统路径处理
	 * @return 按行序排列的标签数组，长度至少为 1
	 * @throws MicaAiException {@link ErrorCode#ILLEGAL_ARGUMENT} 路径为空或文件无有效标签；
	 *                         {@link ErrorCode#NOT_FOUND} 文件不存在；
	 *                         {@link ErrorCode#UNKNOWN} 读取失败
	 */
	public static String[] readLabels(String labelsPath) {
		if (labelsPath == null || labelsPath.isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"labelsPath 必填：分类模型必须与 label_list 成对配置");
		}
		List<String> lines;
		if (OnnxModelSession.isClasspath(labelsPath)) {
			String resource = OnnxModelSession.stripClasspathPrefix(labelsPath);
			lines = IOUtil.readAllLines(
				OnnxModelSession.loadClasspathBytes(resource), StandardCharsets.UTF_8);
		} else {
			Path path = Paths.get(labelsPath);
			if (!Files.exists(path)) {
				throw new MicaAiException(ErrorCode.NOT_FOUND,
					"label_list 文件不存在: " + labelsPath);
			}
			try {
				lines = Files.readAllLines(path, StandardCharsets.UTF_8);
			} catch (IOException e) {
				throw new MicaAiException(ErrorCode.UNKNOWN,
					"读取 label_list 失败: " + labelsPath, e);
			}
		}
		List<String> labels = new ArrayList<>(lines.size());
		for (String line : lines) {
			String label = line == null ? "" : line.trim();
			if (!label.isEmpty()) {
				labels.add(label);
			}
		}
		if (labels.isEmpty()) {
			throw new MicaAiException(ErrorCode.ILLEGAL_ARGUMENT,
				"label_list 为空: " + labelsPath);
		}
		return labels.toArray(new String[0]);
	}
}
