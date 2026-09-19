# Changelog

本项目的所有显著变更都会记录在此文件中。版本遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范。

## [1.0.0] - 2026-09-19

🎉 **首个正式版发布** —— mica-ai 1.0.0 正式发布，提供四个可商用、零 Python、纯 ONNX Runtime 的 Java 8+ AI 能力：

### ✨ 新增能力

- 🎭 `mica-ai-face` —— OpenCV Zoo 人脸识别：YuNet 检测（640×640 BGR anchor-free + 5 关键点）+ SFace 128d Embedding（L2 归一化）+ MiniFASNetV2 活体（real/print/replay 三分类）+ 头像提取（自动摆正 / 分块兜底 / 残角精修）+ 证件卡片提取（掩膜 + 四边形拟合 + 透视矫正 + USM/CLAHE）
- 📄 `mica-ai-filetype` —— Google Magika `standard_v3_3` Java 复刻，214 类文件类型识别；包含 HIGH_CONFIDENCE / MEDIUM_CONFIDENCE / BEST_GUESS 三种预测模式
- 🚗 `mica-ai-plate` —— HyperLPR3 v20230229 中国车牌识别：检测（320/640）+ CRNN 字符识别（77 token CTC）+ 颜色分类（黄/蓝/绿）+ 10 类判型（含港澳 / 学 / WJ 等规则）
- 📐 `mica-ai-layout` —— PaddleOCR PP-DocLayoutV3 文档版面分析（25 类 + V3 reading order）；模型 125MB 不随仓库分发，需本地放模型

### 🔌 Spring Boot Starter

- `mica-ai-face-spring-boot-starter`（`mica.ai.face` 前缀）
- `mica-ai-filetype-spring-boot-starter`（`mica.ai.filetype` 前缀）
- `mica-ai-plate-spring-boot-starter`（`mica.ai.plate` 前缀）
- `mica-ai-layout-spring-boot-starter`（`mica.ai.layout` 前缀）

全部基于 `mica-auto` 生成 `META-INF/spring.factories` / `AutoConfiguration.imports`，零配置即可注入引擎 Bean，`@ConditionalOnMissingBean` 支持自定义实现替换。

### 🧱 基础模块

- `mica-ai-common` —— 跨能力共享基础设施：`OnnxModelSession`（含 `classpath:` 前缀资源解析）、`OrtSessionFactory`（统一构造 OrtSessionOptions）、`OrtProviders`（CPU/CoreML/CUDA 自动选择与回退）、`MicaAiException` / `ErrorCode` 统一异常体系

### 📦 模型与构建

- 模型资产 `model-tools/` 全部直接入库（YuNet + SFace + MiniFASNetV2 + Magika standard_v3_3 + HyperLPR3 v20230229）
- 唯一例外：PP-DocLayoutV3 模型 125MB 超出「<50MB 入库」约定，已在 `.gitignore` 显式排除；`LayoutIntegrationTest` 在模型缺失时用 `Assumptions` 整类跳过
- Maven 顶层 `${revision}=1.0.0`，JDK 8+ / Spring Boot 2.7.x / ONNX Runtime 1.18.0 / OpenCV 4.9.0（openpnp）

### ✅ License 一句话

代码 Apache-2.0；所有依赖模型 **全部可商用**：YuNet + SFace（Apache-2.0）、MiniFASNetV2（MIT）、Magika `standard_v3_3`（Apache-2.0）、HyperLPR3 v20230229（Apache-2.0）、PP-DocLayoutV3（Apache-2.0）。

---

## 历史

### [2026.06.01] - 2026-09-15

集成开发期快照：

- `mica-ai-face` 主线能力就位（YuNet + SFace + MiniFASNetV2 + 头像 / 卡片）
- `mica-ai-filetype` 完成 Magika 复刻
- `mica-ai-face-spring-boot-starter` / `mica-ai-filetype-spring-boot-starter` 自动装配落地
- `mica-ai-common` 抽出 ONNX 通用基础设施 + `MicaAiException`
- 移除 `model-tools/` Python 工具链（模型已直接入库）
- 移除 `javax.annotation` 依赖，统一 ONNX 配置到 `mica-ai-common`