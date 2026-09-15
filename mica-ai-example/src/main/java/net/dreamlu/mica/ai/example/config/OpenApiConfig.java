/*
 * Copyright (c) 2024-2026 mica-ai
 */
package net.dreamlu.mica.ai.example.config;

import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * OpenAPI 3 / Swagger UI 元信息。
 */
@Configuration
public class OpenApiConfig {

	@Bean
	public OpenAPI micaAiOpenApi() {
		return new OpenAPI()
			.info(new Info()
				.title("mica-ai Example API")
				.description("mica-ai 人脸能力（YuNet + SFace + MiniFASNetV2）的 Spring Boot 集成 Demo。"
					+ "模型路径在 application.yml 里统一配置。")
				.version("1.0.0")
				.contact(new Contact()
					.name("mica-ai")
					.url("https://www.dreamlu.net"))
				.license(new License()
					.name("Apache License 2.0")
					.url("https://www.apache.org/licenses/LICENSE-2.0")))
			.components(new Components());
	}
}