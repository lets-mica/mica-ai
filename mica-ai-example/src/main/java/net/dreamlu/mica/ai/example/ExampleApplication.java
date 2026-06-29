/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * mica-ai Spring Boot 集成测试 / Demo 启动入口。
 *
 * <p>本应用聚合了 mica-ai 全部 6 个 Starter：
 * <ul>
 *     <li>mica-ai-tts：Kokoro 语音合成</li>
 *     <li>mica-ai-voice：SenseVoice 语音识别</li>
 *     <li>mica-ai-ppocr：PP-OCRv6 文字识别</li>
 *     <li>mica-ai-speaker：ERes2Net 声纹识别</li>
 *     <li>mica-ai-intent：BERT 中文意图分类</li>
 *     <li>mica-ai-face：OpenCV Zoo YuNet + SFace 人脸识别</li>
 * </ul>
 *
 * <p>启动前需在 {@code src/main/resources/application.yml} 中至少启用一个能力并配置模型路径，
 * 未配置的能力会在启动时 fail-fast（启用前提下）。
 */
@SpringBootApplication
public class ExampleApplication {

	public static void main(String[] args) {
		SpringApplication.run(ExampleApplication.class, args);
	}
}