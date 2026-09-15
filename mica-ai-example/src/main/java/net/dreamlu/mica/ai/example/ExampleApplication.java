/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * mica-ai Spring Boot 集成测试 / Demo 启动入口。
 *
 * <p>当前 Starter：
 * <ul>
 *     <li>mica-ai-face：OpenCV Zoo YuNet + SFace 人脸识别</li>
 * </ul>
 *
 * <p>启动前需在 {@code src/main/resources/application.yml} 中启用 face 并配置模型路径，
 * 未配置模型路径会在启动时 fail-fast（enabled=true 前提下）。
 */
@SpringBootApplication
public class ExampleApplication {

	public static void main(String[] args) {
		SpringApplication.run(ExampleApplication.class, args);
	}
}