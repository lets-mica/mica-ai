# mica-ai-starters

> mica-ai Spring Boot Starter 聚合模块，一行配置即可将 AI 能力接入 Spring Boot 应用。

本模块为聚合 POM（packaging=pom），包含以下 4 个 starter：

---

## Starter 一览

| Starter | 功能 | 配置前缀 |
|---------|------|---------|
| [mica-ai-ppocr-spring-boot-starter](mica-ai-ppocr-spring-boot-starter/README.md) | PP-OCRv6 文字识别 | `mica.ai.ppocr` |
| [mica-ai-tts-spring-boot-starter](mica-ai-tts-spring-boot-starter/README.md) | Kokoro TTS 语音合成 | `mica.ai.tts` |
| [mica-ai-voice-spring-boot-starter](mica-ai-voice-spring-boot-starter/README.md) | SenseVoice 语音识别 | `mica.ai.voice` |
| [mica-ai-speaker-spring-boot-starter](mica-ai-speaker-spring-boot-starter/README.md) | ERes2Net 声纹识别 | `mica.ai.speaker` |

---

## 统一设计

所有 starter 遵循一致的设计模式：

- **AutoConfiguration** — 条件装配核心引擎 Bean（`@ConditionalOnClass` + `@ConditionalOnProperty`）
- **Properties** — `@ConfigurationProperties` 绑定配置文件，使用 Lombok `@Data` 注解
- **mica-auto** — 自动生成 `META-INF/spring/...AutoConfiguration.imports`，无需手动维护
- **生命周期** — Bean 设置 `destroyMethod = "close"`，应用关闭时自动释放 ONNX 资源

### Maven 依赖

所有 starter 的 groupId 为 `net.dreamlu`，artifactId 以 `-spring-boot-starter` 结尾：

```xml
<!-- 以 TTS 为例 -->
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-tts-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

引入后无需任何 Java 代码，仅需在 `application.yml` 中配置模型路径即可自动注入对应 Bean。
