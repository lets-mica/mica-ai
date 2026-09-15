# mica-ai-example

> mica-ai Spring Boot 集成测试 / Demo：聚合 Face + Filetype Starter 的可运行示例与 `@SpringBootTest`。

本模块面向集成测试场景：

1. **可运行的 Spring Boot Demo**：`mvn spring-boot:run`（`develop` profile）启动后，可通过 REST 端点触发 face / filetype 能力。
2. **`@SpringBootTest` 集成测试**：覆盖 Starter 的自动装配行为（`enabled` 开关、fail-fast、Bean 注入）。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 8+ | 推荐 Temurin / Azul Zulu 8、11、17 |
| Maven | 3.6+ | 多模块构建 |
| Spring Boot | 2.7.x | 由根 BOM 引入 |
| ONNX 模型 | YuNet + SFace + Magika standard_v3_3 | 见 [mica-ai-face/README.md](../mica-ai-core/mica-ai-face/README.md) / [mica-ai-filetype/README.md](../mica-ai-core/mica-ai-filetype/README.md) |

---

## 2. 快速开始

### 2.1 默认启动

```bash
mvn -pl mica-ai-example -am -DskipTests clean install
mvn -pl mica-ai-example spring-boot:run
```

启动后访问 `http://localhost:8181`。

### 2.2 启用 face / filetype

编辑 [src/main/resources/application.yml](src/main/resources/application.yml)，把模型路径改成你的实际路径（默认走 classpath）：

```yaml
mica:
  ai:
    face:
      enabled: true
      model:
        detection:
          path: E:/codes/ai/mica-ai/model-tools/face/model/out/face_detection_yunet_2023mar.onnx
        recognition:
          path: E:/codes/ai/mica-ai/model-tools/face/model/out/face_recognition_sface_2021dec.onnx
        liveness:
          path: E:/codes/ai/mica-ai/model-tools/face/model/out/2.7_80x80_MiniFASNetV2.onnx
    filetype:
      enabled: true
      model-path: E:/codes/ai/mica-ai/model-tools/filetype/model/out/model.onnx
      config-path: E:/codes/ai/mica-ai/model-tools/filetype/model/out/config.min.json
      content-types-path: E:/codes/ai/mica-ai/model-tools/filetype/model/out/content_types_kb.min.json
```

> ⚠️ 当 `enabled=true` 但必填项缺失时，应用启动会 **fail-fast** 抛出 `MicaAiException`。
> 不希望加载该能力时，设 `enabled: false` 即可。

### 2.3 跑集成测试

```bash
mvn -pl mica-ai-example -am test
```

测试覆盖 `ExampleApplicationContextTest`，验证 Spring Boot 上下文能正常加载 Starter Bean。

> 集成测试不依赖任何真实 ONNX 模型文件，可放心在 CI 中执行。

---

## 3. REST 端点

| 路径 | 方法 | 描述 |
|------|------|------|
| `/face/detect` | POST multipart | 上传图片，返回人脸框 + 关键点 + score |
| `/face/extract` | POST multipart | 上传图片，返回 128d Embedding + L2 norm |
| `/filetype/detect` | POST multipart | 上传任意文件，返回 modelLabel / outputLabel / score / mimeType / group / description |
| `/filetype/detect-bytes` | POST octet-stream | 上传二进制，返回同上字段 |

Swagger UI：` http://localhost:8181/swagger-ui.html`

---

## 4. 项目结构

```
mica-ai-example/
├── pom.xml                                          # face + filetype starter + spring-boot-starter-web
├── src/main/java/net/dreamlu/mica/ai/example/
│   ├── ExampleApplication.java                      # @SpringBootApplication 入口
│   ├── config/OpenApiConfig.java
│   └── controller/
│       ├── FaceController.java                      # /face/detect + /face/extract
│       └── FiletypeController.java                  # /filetype/detect + /filetype/detect-bytes
├── src/main/resources/
│   ├── application.yml                              # face + filetype 示例配置
│   └── logback-spring.xml
└── src/test/java/net/dreamlu/mica/ai/example/
    └── ExampleApplicationContextTest.java           # @SpringBootTest 集成测试
```

---

## 5. 已知约束

- **multipart 上传**：默认最大 50MB / 请求 100MB，可在 `application.yml` 中调整 `spring.servlet.multipart`。