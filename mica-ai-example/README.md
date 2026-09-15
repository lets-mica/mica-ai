# mica-ai-example

> mica-ai Spring Boot 集成测试 / Demo：聚合 Face Starter 的可运行示例与 `@SpringBootTest`。

本模块面向集成测试场景：

1. **可运行的 Spring Boot Demo**：`mvn spring-boot:run` 启动后，可通过 REST 端点触发 face 能力。
2. **`ApplicationContextRunner` 集成测试**：覆盖 Face Starter 的自动装配行为（`enabled` 开关、fail-fast、Bean 注入）。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Azul Zulu 17 / Temurin 17 |
| Maven | 3.6+ | 多模块构建 |
| Spring Boot | 4.1.0+ | 由根 BOM 引入 |
| ONNX 模型 | YuNet + SFace | 见 [mica-ai-face/README.md](../mica-ai-core/mica-ai-face/README.md) |

---

## 2. 快速开始

### 2.1 默认启动（face disabled）

```bash
mvn -pl mica-ai-example -am -DskipTests clean install
mvn -pl mica-ai-example spring-boot:run
```

启动后访问 `http://localhost:8181`，由于 face 能力未启用，Controller 已被 `@ConditionalOnProperty(mica.ai.face.enabled)` 跳过注册。

### 2.2 启用 face

编辑 [src/main/resources/application.yml](src/main/resources/application.yml)，把模型路径改成你的实际路径：

```yaml
mica:
  ai:
    face:
      enabled: true
      det-model-path: E:/codes/ai/mica-ai/model-tools/models/face/face_detection_yunet_2023mar.onnx
      rec-model-path: E:/codes/ai/mica-ai/model-tools/models/face/face_recognition_sface_2021dec.onnx
```

> ⚠️ 当 `enabled=true` 但必填项（如 `det-model-path`）缺失时，应用启动会 **fail-fast** 抛出 `MicaAiException`。
> 不希望加载该能力时，设 `enabled: false` 即可。

### 2.3 跑集成测试

```bash
mvn -pl mica-ai-example -am test
```

测试覆盖：

- `disabledAllStartClean`：默认配置下 ApplicationContext 能正常启动，FaceEngine Bean 不存在。
- `faceEnabledButMissingRequiredShouldFailFast`：face `enabled=true` 但缺 `det-model-path` → 启动抛 `MicaAiException`。

> 集成测试不依赖任何真实 ONNX 模型文件，可放心在 CI 中执行。

---

## 3. 在你的应用里注入 Engine

`mica-ai-example` 当前只演示 Starter 自动装配，不提供 REST Controller。在你自己的业务代码里：

```java
@Service
@RequiredArgsConstructor
public class FaceService {
    private final FaceEngine faceEngine;   // 由 FaceAutoConfiguration 自动注入

    public List<FaceEmbedding> extract(Path imagePath) {
        return faceEngine.extract(imagePath);
    }
}
```

> Controller 层可自行用 `@ConditionalOnProperty(prefix = "mica.ai.face", name = "enabled")` 装饰，未启用时不注册对应 Bean。

---

## 4. 项目结构

```
mica-ai-example/
├── pom.xml                                          # 依赖 face-spring-boot-starter + spring-boot-starter-web
├── src/main/java/net/dreamlu/mica/ai/example/
│   └── ExampleApplication.java                      # @SpringBootApplication 入口
├── src/main/resources/
│   ├── application.yml                              # face disabled + 启用示例
│   └── logback-spring.xml
└── src/test/java/net/dreamlu/mica/ai/example/
    └── ExampleApplicationContextTest.java           # ApplicationContextRunner 集成测试
```

---

## 5. 已知约束

- **multipart 上传**：默认最大 50MB / 请求 100MB，可在 `application.yml` 中调整 `spring.servlet.multipart`。
