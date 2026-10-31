# mica-ai-starters

> mica-ai Spring Boot Starter 聚合模块，一行配置即可将 AI 能力接入 Spring Boot 应用。

本模块为聚合 POM（`packaging=pom`），包含 **13 个 starter**：

| Starter | 功能 | 配置前缀 |
|---------|------|---------|
| [mica-ai-face-spring-boot-starter](mica-ai-face-spring-boot-starter/README.md) | 人脸检测 + 128d 特征 + 活体 + 头像 / 证件卡片提取 + 质量评估 + 表情识别 | `mica.ai.face` |
| [mica-ai-filetype-spring-boot-starter](mica-ai-filetype-spring-boot-starter/README.md) | Google Magika 复刻，214 类文件类型识别（含 / 排除置信度三模式） | `mica.ai.filetype` |
| [mica-ai-plate-spring-boot-starter](mica-ai-plate-spring-boot-starter/README.md) | HyperLPR3 中国车牌识别（检测 + CRNN 识别 + 颜色分类 + 10 类判型） | `mica.ai.plate` |
| [mica-ai-layout-spring-boot-starter](mica-ai-layout-spring-boot-starter/README.md) | PP-DocLayoutV3 文档版面分析（25 类 + V3 阅读顺序；模型 125MB 不随仓库分发，需本地放模型） | `mica.ai.layout` |
| [mica-ai-matting-spring-boot-starter](mica-ai-matting-spring-boot-starter/README.md) | U²-Net 族通用抠图（透明底 / 纯色底 / 二值掩码；`u2netp` 4.36MB 入库，`u2net` / `u2net_human_seg` 走 `model-path` 外置切换） | `mica.ai.matting` |
| [mica-ai-humanseg-spring-boot-starter](mica-ai-humanseg-spring-boot-starter/README.md) | PP-HumanSeg V2-Lite 人像分割（透明底 / 纯色底 / 二值掩码；`192×192` 5.88MB 入库，竖版同族走 `model-path` 外置切换） | `mica.ai.humanseg` |
| [mica-ai-detect-spring-boot-starter](mica-ai-detect-spring-boot-starter/README.md) | NanoDet-Plus / YOLOX-s 通用目标检测（COCO 80 类；两种模型族切换只改 `family` + `input-size`） | `mica.ai.detect` |
| [mica-ai-track-spring-boot-starter](mica-ai-track-spring-boot-starter/README.md) | ByteTrack 多目标跟踪（纯算法零模型，输入任意检测框序列，输出稳定 track id） | `mica.ai.track` |
| [mica-ai-docalign-spring-boot-starter](mica-ai-docalign-spring-boot-starter/README.md) | DocAligner 文档角点检测 + 透视矫正（4 角点 + 一键拉平；4.55MB 入库，fastvit 同族走 `model-path` 外置切换） | `mica.ai.docalign` |
| [mica-ai-enhance-spring-boot-starter](mica-ai-enhance-spring-boot-starter/README.md) | Real-ESRGAN 4 倍图像超分（2.33MB 入库；大图自动 tile 切分，`tile-size` / `tile-overlap` 可调） | `mica.ai.enhance` |
| [mica-ai-textline-spring-boot-starter](mica-ai-textline-spring-boot-starter/README.md) | PP-LCNet 文本行方向分类（0° / 180° 判定 + 转正；`x1_0` 6.46MB 入库，`x0_25` 轻量版走 `model-path` 外置切换） | `mica.ai.textline` |
| [mica-ai-classify-spring-boot-starter](mica-ai-classify-spring-boot-starter/README.md) | PaddleClas PULC 通用图像分类（整图方向 / 清晰度 / 人物存在 / 表格 6 属性；`labels-path` 必填，多模型分别声明 Bean） | `mica.ai.classify` |
| [mica-ai-captcha-spring-boot-starter](mica-ai-captcha-spring-boot-starter/README.md) | ddddocr 通用验证码识别（CRNN + CTC，8210 类字符集；**模型与字符集外置配置**，`model-path` / `charset-path` 必填） | `mica.ai.captcha` |

---

## 统一设计

所有 starter 遵循一致的设计模式：

- **AutoConfiguration** — 条件装配核心引擎 Bean（`@ConditionalOnClass` + `@ConditionalOnProperty`）
- **Properties** — `@ConfigurationProperties` 绑定配置文件
- **mica-auto** — 自动生成 `META-INF/spring/...AutoConfiguration.imports`，无需手动维护
- **生命周期** — Bean 设置 `destroyMethod = "close"`，应用关闭时自动释放 ONNX 资源
- **fail-fast** — `enabled=true` 但必填配置缺失时启动直接抛 `MicaAiException`

---

## Maven 依赖（按需引入）

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

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-layout-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-matting-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-humanseg-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-detect-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-track-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-docalign-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-enhance-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-textline-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-classify-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-captcha-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

引入后无需任何 Java 代码，仅需在 `application.yml` 中配置模型路径即可自动注入对应 Bean。

> ⚠️ 模型随仓库分发但**不打包进 jar**，各能力的 `model-path` **必填**；未配置时启动即 fail-fast，并在错误消息里指向 `model-tools/<cap>/models/` 下的文件。用户自己把模型放进 classpath 时，路径写 `classpath:` 前缀即可。

完整可运行示例见 [`mica-ai-example`](../mica-ai-example/README.md)。
