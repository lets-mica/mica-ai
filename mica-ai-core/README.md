# mica-ai-core

> mica-ai 核心能力聚合模块，**零 Spring 依赖**，纯 Java 实现。

本模块为聚合 POM（`packaging=pom`），包含 3 个能力子模块：

| 模块 | 功能 | 模型 |
|------|------|------|
| [mica-ai-face](mica-ai-face/README.md) | OpenCV Zoo 人脸识别（YuNet 检测 + 5 关键点对齐 + SFace 128d Embedding + MiniFASNetV2 活体 + 头像 / 证件卡片提取） | YuNet / SFace / MiniFASNetV2 |
| [mica-ai-filetype](mica-ai-filetype/README.md) | Google Magika 文件类型识别（214 类，含 / 排除置信度三模式） | Magika `standard_v3_3` |
| [mica-ai-plate](mica-ai-plate/README.md) | HyperLPR3 中国车牌识别（检测 + CRNN 字符识别 + 颜色分类 + 10 类判型） | HyperLPR3 v20230229 |

子模块依赖 [`mica-ai-common`](../mica-ai-common/)，共享 `MicaAiException` 统一异常、`OnnxModelSession` 单模型会话持有者、`OrtSessionFactory` ONNX 会话工厂。

---

## 使用入口

```java
// 纯 Java，无需 Spring
try (ModelManager m = ModelManager.create(config)) {
    new FaceDetector(m).detect(image);
}
try (FiletypeDetector d = new FiletypeDetector(fc)) {
    d.detectPath(path);
}
try (PlatePipeline p = PlatePipeline.create(pc)) {
    p.recognizePath(path);
}
```

各能力的 Spring Boot 接入见 [`mica-ai-starters`](../mica-ai-starters/README.md)。