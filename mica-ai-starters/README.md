# mica-ai-starters

> mica-ai Spring Boot Starter 聚合模块，一行配置即可将 AI 能力接入 Spring Boot 应用。

本模块为聚合 POM（`packaging=pom`），包含 3 个 starter：

| Starter | 功能 | 配置前缀 |
|---------|------|---------|
| [mica-ai-face-spring-boot-starter](mica-ai-face-spring-boot-starter/README.md) | OpenCV Zoo 人脸识别（YuNet 检测 + SFace 128d 向量 + MiniFASNetV2 活体 + 头像 / 证件卡片提取） | `mica.ai.face` |
| [mica-ai-filetype-spring-boot-starter](mica-ai-filetype-spring-boot-starter/README.md) | Google Magika 文件类型识别（214 类，三种置信度模式） | `mica.ai.filetype` |
| [mica-ai-plate-spring-boot-starter](mica-ai-plate-spring-boot-starter/README.md) | HyperLPR3 中国车牌识别（检测 + 识别 + 颜色分类 + 10 类判型） | `mica.ai.plate` |

---

## 统一设计

所有 starter 遵循一致的设计模式：

- **AutoConfiguration** — 条件装配核心引擎 Bean（`@ConditionalOnClass` + `@ConditionalOnProperty`）
- **Properties** — `@ConfigurationProperties` 绑定配置文件
- **mica-auto** — 自动生成 `META-INF/spring/...AutoConfiguration.imports`，无需手动维护
- **生命周期** — Bean 设置 `destroyMethod = "close"`，应用关闭时自动释放 ONNX 资源
- **fail-fast** — `enabled=true` 但必填配置缺失时启动直接抛 `MicaAiException`

### Maven 依赖（按需引入）

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-face-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-filetype-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-plate-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

引入后无需任何 Java 代码，仅需在 `application.yml` 中配置模型路径即可自动注入对应 Bean。

完整可运行示例见 [`mica-ai-example`](../mica-ai-example/README.md)。