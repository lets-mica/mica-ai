# mica-ai-core

> mica-ai 核心能力聚合模块，**零 Spring 依赖**，纯 Java 实现。

本模块为聚合 POM（`packaging=pom`），包含 **13 个能力子模块** + 1 个公共基础模块：

| 模块 | 功能 | 模型 |
|------|------|------|
| [mica-ai-face](mica-ai-face/README.md) | OpenCV Zoo 人脸识别（YuNet 检测 + 5 关键点对齐 + SFace 128d Embedding + MiniFASNetV2 活体 + 头像 / 证件卡片提取 + eDifFIQA 质量评估 + 7 类表情识别） | YuNet / SFace / MiniFASNetV2 / eDifFIQA / MobileFaceNet |
| [mica-ai-filetype](mica-ai-filetype/README.md) | Google Magika 文件类型识别（214 类，含 / 排除置信度三模式） | Magika `standard_v3_3` |
| [mica-ai-plate](mica-ai-plate/README.md) | HyperLPR3 中国车牌识别（检测 + CRNN 字符识别 + 颜色分类 + 10 类判型） | HyperLPR3 v20230229 |
| [mica-ai-layout](mica-ai-layout/README.md) | PP-DocLayoutV3 文档版面分析（25 类 + V3 阅读顺序） | PP-DocLayoutV3（125MB，不随仓库分发） |
| [mica-ai-matting](mica-ai-matting/README.md) | U²-Net 族通用抠图（原尺寸 alpha 掩码 / 透明底 / 纯色底 / 二值） | U²-Net `u2netp` |
| [mica-ai-humanseg](mica-ai-humanseg/README.md) | PP-HumanSeg V2-Lite 人像分割（原尺寸人像 alpha 掩码） | PP-HumanSeg V2-Lite |
| [mica-ai-detect](mica-ai-detect/README.md) | 通用目标检测（COCO 80 类，NanoDet-Plus / YOLOX-s 两族） | NanoDet-Plus / YOLOX-s |
| [mica-ai-track](mica-ai-track/README.md) | ByteTrack 多目标跟踪（**纯算法零模型**，稳定 track id） | 无 |
| [mica-ai-docalign](mica-ai-docalign/README.md) | DocAligner 文档角点检测 + 透视矫正（4 角点 + 一键拉平） | DocAligner `lcnet100` |
| [mica-ai-enhance](mica-ai-enhance/README.md) | Real-ESRGAN 图像超分（4 倍放大，大图 tile 推理） | Real-ESRGAN `x4v3` |
| [mica-ai-textline](mica-ai-textline/README.md) | PP-LCNet 文本行方向分类（0° / 180° 判定 + 转正） | PP-LCNet `x1_0` |
| [mica-ai-classify](mica-ai-classify/README.md) | PaddleClas PULC 通用图像分类（Top-K / 多标签） | PULC × 4 |
| [mica-ai-captcha](mica-ai-captcha/README.md) | ddddocr 通用验证码识别（CRNN + CTC，8210 类字符集） | ddddocr CRNN |
| [mica-ai-common](mica-ai-common/README.md) | 跨能力公共基础设施（`MicaAiException` 统一异常、`OnnxModelSession`、`OrtSessionFactory`、`IOUtil`） | 无 |

子模块依赖 [`mica-ai-common`](mica-ai-common/README.md)，共享 `MicaAiException` 统一异常、`OnnxModelSession` 单模型会话持有者、`OrtSessionFactory` ONNX 会话工厂。

---

## 使用入口

```java
// 纯 Java，无需 Spring
try (ModelManager m = ModelManager.create(config)) {
    new FaceDetector(m).detect(image);
}
try (FiletypeDetector d = FiletypeDetector.create(fc)) {
    d.detectPath(path);
}
try (PlatePipeline p = PlatePipeline.create(pc)) {
    p.recognizePath(path);
}
try (DetectEngine e = DetectEngine.create(dc)) {
    e.detectPath(path);
}
try (ByteTracker t = ByteTracker.create(tc)) {
    t.update(detections);
}
```

各能力的 Spring Boot 接入见 [`mica-ai-starters`](../mica-ai-starters/README.md)。

---

## 两条硬约束

1. **零 Spring 依赖** —— core 模块不引任何 `spring-*`，可在非 Spring 环境直接使用
2. **模型路径必填** —— 模型随仓库分发但**不打包进 jar**，各能力的 `modelPath` 必填；未配置时启动即 fail-fast 并提示指向 `model-tools/<cap>/models/` 下的文件。用户显式配置的路径仍支持 `classpath:` 前缀
