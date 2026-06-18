# mica-ai-core

> mica-ai 核心能力聚合模块，零 Spring 依赖，纯 Java 实现。

本模块为聚合 POM（packaging=pom），包含以下 5 个子模块：

---

## 子模块一览

| 模块 | 功能 | 技术栈 |
|------|------|--------|
| [mica-ai-ppocr](mica-ai-ppocr/README.md) | PP-OCRv6 文字识别 | ONNX Runtime + OpenCV + JTS |
| [mica-ai-tts](mica-ai-tts/README.md) | Kokoro TTS 语音合成 | ONNX Runtime + houbb/pinyin (可选) |
| [mica-ai-voice](mica-ai-voice/README.md) | SenseVoice 语音识别 | ONNX Runtime + 纯 Java Mel/FFT |
| [mica-ai-speaker](mica-ai-speaker/README.md) | ERes2Net 声纹识别 | ONNX Runtime + JTransforms FFT |
| [mica-ai-intent](mica-ai-intent/README.md) | BERT 中文意图识别 | ONNX Runtime + 手写 BERT 分词 |

所有子模块均依赖 [mica-ai-common](../mica-ai-common/README.md)，共享 `MicaAiException` 统一异常、`OrtProviders` Provider 管理和 `AudioUtils` 音频工具。

---

## 设计原则

- **零 Spring 依赖** — 核心模块不引入任何 Spring 类，可在任何 Java 17+ 项目中独立使用
- **Builder 模式** — 所有引擎均通过 Builder 风格的 Config 类构造，链式调用清晰
- **AutoCloseable** — 所有引擎实现 `AutoCloseable`，支持 try-with-resources 自动释放 ONNX 资源
- **可插拔** — 关键组件（如 TTS 的 G2P）通过接口注入，支持自定义实现
