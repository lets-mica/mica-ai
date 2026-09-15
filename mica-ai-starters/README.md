# mica-ai-starters

> mica-ai Spring Boot Starter 聚合模块，一行配置即可将 AI 能力接入 Spring Boot 应用。

本模块为聚合 POM（packaging=pom），包含 1 个 starter：

---

## Starter 一览

| Starter | 功能 | 配置前缀 |
|---------|------|---------|
| [mica-ai-face-spring-boot-starter](mica-ai-face-spring-boot-starter/README.md) | OpenCV Zoo 人脸识别（YuNet 检测 + SFace 512d 向量） | `mica.ai.face` |

---

## 统一设计

所有 starter 遵循一致的设计模式：

- **AutoConfiguration** — 条件装配核心引擎 Bean（`@ConditionalOnClass` + `@ConditionalOnProperty`）
- **Properties** — `@ConfigurationProperties` 绑定配置文件，使用 Lombok `@Data` 注解
- **mica-auto** — 自动生成 `META-INF/spring/...AutoConfiguration.imports`，无需手动维护
- **生命周期** — Bean 设置 `destroyMethod = "close"`，应用关闭时自动释放 ONNX 资源

### Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-face-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

引入后无需任何 Java 代码，仅需在 `application.yml` 中配置模型路径即可自动注入对应 Bean。
