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
		System.out.println("http://localhost:8181/swagger-ui.html");
	}
}
