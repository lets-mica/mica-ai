/*
 * Copyright (c) 2019-2029, DreamLu 卢春梦 (596392912@qq.com & dreamlu.net).
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
package net.dreamlu.mica.ai.example;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * 示例页脚本的静态守卫。
 *
 * <p>示例页只有一份 {@code index.html}，没有任何编译期检查：能力卡片里写错一个标识符
 * （例如把 {@code INSPECTION_SAMPLE} 写成 {@code INSECTION_SAMPLE}）会让整页脚本在
 * 解析阶段就抛 {@code ReferenceError}，**所有示例一起加载不出来**，而且报错发生在浏览器里，
 * 构建期完全看不见。
 *
 * <p>本类用一条最朴素的规则挡住这类错误：能力卡片里引用的每一个标识符，要么在脚本里有
 * 声明，要么是 JS 内置全局。
 */
class ExamplePageScriptTest {

	/** 示例页脚本里的能力卡片数组 */
	private static final List<String> CAPS_ARRAYS = Arrays.asList(
		"FACE_CAPS", "DOC_CAPS", "IMAGE_CAPS", "DETECT_CAPS", "MISC_CAPS", "SCENARIO_CAPS");

	/** 允许出现在能力卡片里的 JS 内置全局 */
	private static final Set<String> GLOBALS = new HashSet<>(Arrays.asList(
		"JSON", "String", "Number", "Boolean", "Object", "Array", "Math", "Date",
		"parseInt", "parseFloat", "isNaN", "Infinity", "NaN"));

	/** JS 关键字 / 字面量 */
	private static final Set<String> KEYWORDS = new HashSet<>(Arrays.asList(
		"const", "let", "var", "function", "return", "if", "else", "for", "while",
		"new", "typeof", "null", "true", "false", "undefined", "this", "of", "in"));

	@Test
	void everyIdentifierUsedInCapabilityCardsIsDeclared() throws IOException {
		String script = scriptBlock();

		Set<String> declared = declaredIdentifiers(script);
		assertThat(declared)
			.as("脚本里应能解析出 const / let / var / function 声明")
			.isNotEmpty();

		Set<String> unresolved = new LinkedHashSet<>();
		for (String caps : CAPS_ARRAYS) {
			String region = capsRegion(script, caps);
			assertThat(region).as("应能找到能力卡片数组 %s", caps).isNotEmpty();
			for (String identifier : identifiersOf(region)) {
				if (declared.contains(identifier) || GLOBALS.contains(identifier)
					|| KEYWORDS.contains(identifier)) {
					continue;
				}
				unresolved.add(caps + " → " + identifier);
			}
		}

		assertThat(unresolved)
			.as("能力卡片里引用的标识符必须在脚本中有声明（写错会让整页脚本抛 ReferenceError）")
			.isEmpty();
	}

	@Test
	void scriptBlockIsFound() throws IOException {
		assertThat(scriptBlock()).as("示例页应包含内联脚本").isNotEmpty();
	}

	/**
	 * 用 node 真正驱动页面脚本，逐张调用能力卡片，校验请求格式与调用过程。
	 *
	 * <p>示例页没有编译期检查，能力卡片一旦与请求格式不匹配（比如带文件的卡片被当成
	 * 「JSON 请求体」发出去），只在浏览器里炸。这里把它变成构建期会红的断言。
	 * 环境没有 node 时整项跳过，不阻塞构建。
	 */
	@Test
	void everyCapabilityCardSendsTheRequestFormatItsFieldsImply() throws Exception {
		Path script = pageCheckScript();
		assertThat(script).as("卡片校验脚本应存在").isNotNull();

		ProcessBuilder builder = new ProcessBuilder("node", script.toString(), page().toString());
		builder.redirectErrorStream(true);
		Process process = builder.start();
		String output;
		try (java.io.InputStream in = process.getInputStream()) {
			output = new String(readAll(in), StandardCharsets.UTF_8);
		}
		int exit = process.waitFor();
		assumeTrue(exit != 127, "环境里没有 node，跳过示例页卡片校验");

		assertThat(output).as("示例页能力卡片校验输出").isNotEmpty();
		assertThat(exit)
			.as("示例页能力卡片校验未通过:\n%s", output)
			.isZero();
	}

	/**
	 * 卡片校验脚本（测试资源）。
	 */
	private static Path pageCheckScript() {
		Path path = Paths.get("src", "test", "resources", "page-check", "card-check.js");
		if (Files.exists(path)) {
			return path;
		}
		return repoFile("mica-ai-example/src/test/resources/page-check/card-check.js");
	}

	private static byte[] readAll(java.io.InputStream in) throws IOException {
		java.io.ByteArrayOutputStream buffer = new java.io.ByteArrayOutputStream();
		byte[] chunk = new byte[4096];
		int read;
		while ((read = in.read(chunk)) > 0) {
			buffer.write(chunk, 0, read);
		}
		return buffer.toByteArray();
	}

	/**
	 * 抽取页面里的内联脚本。
	 */
	private static String scriptBlock() throws IOException {
		String html = new String(Files.readAllBytes(page()), StandardCharsets.UTF_8);
		Matcher matcher = Pattern.compile("<script[^>]*>(.*?)</script>", Pattern.DOTALL).matcher(html);
		assertThat(matcher.find()).as("示例页应包含内联 <script> 块").isTrue();
		return matcher.group(1);
	}

	private static Path page() {
		Path path = repoFile("mica-ai-example/src/main/resources/static/index.html");
		assertThat(path).as("示例页应存在: mica-ai-example/src/main/resources/static/index.html").isNotNull();
		return path;
	}

	/**
	 * 从当前目录逐级向上找仓库里的文件（surefire 的工作目录不固定）。
	 */
	private static Path repoFile(String relative) {
		Path dir = Paths.get("").toAbsolutePath();
		for (int i = 0; i < 6 && dir != null; i++) {
			Path candidate = dir.resolve(relative);
			if (Files.exists(candidate)) {
				return candidate;
			}
			dir = dir.getParent();
		}
		return null;
	}

	/**
	 * 取某个能力卡片数组的字面量区域（从 {@code const X = [} 到该数组结束的 {@code ];}）。
	 */
	private static String capsRegion(String script, String caps) {
		int start = script.indexOf("const " + caps + " = [");
		if (start < 0) {
			return "";
		}
		int end = script.indexOf("\n];", start);
		return end < 0 ? script.substring(start) : script.substring(start, end);
	}

	/**
	 * 脚本里声明过的标识符。
	 */
	private static Set<String> declaredIdentifiers(String script) {
		Set<String> names = new LinkedHashSet<>();
		Matcher matcher = Pattern.compile("\\b(?:const|let|var|function)\\s+([A-Za-z_$][A-Za-z0-9_$]*)")
			.matcher(script);
		while (matcher.find()) {
			names.add(matcher.group(1));
		}
		return names;
	}

	/**
	 * 区域里出现的裸标识符（剔除字符串字面量、对象键、属性访问与数字）。
	 */
	private static Set<String> identifiersOf(String region) {
		String stripped = region
			// 去掉字符串字面量
			.replaceAll("'(?:[^'\\\\]|\\\\.)*'", " ")
			.replaceAll("\"(?:[^\"\\\\]|\\\\.)*\"", " ")
			// 去掉注释
			.replaceAll("//[^\\n]*", " ")
			.replaceAll("/\\*.*?\\*/", " ");
		Set<String> names = new LinkedHashSet<>();
		Matcher matcher = Pattern.compile("([A-Za-z_$][A-Za-z0-9_$]*)").matcher(stripped);
		while (matcher.find()) {
			String name = matcher.group(1);
			int end = matcher.end();
			// 属性访问：前面是 '.' → 属于对象成员，不是标识符引用
			int before = matcher.start() - 1;
			while (before >= 0 && Character.isWhitespace(stripped.charAt(before))) {
				before--;
			}
			if (before >= 0 && stripped.charAt(before) == '.') {
				continue;
			}
			// 对象字面量的键：后面是 ':' → 是属性名，不是引用
			int after = end;
			while (after < stripped.length() && Character.isWhitespace(stripped.charAt(after))) {
				after++;
			}
			if (after < stripped.length() && stripped.charAt(after) == ':') {
				continue;
			}
			names.add(name);
		}
		return names;
	}

}
