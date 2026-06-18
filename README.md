# Mica AI

> 面向 Java 生态，提供统一、简洁、可扩展的 API，让 AI 能力开箱即用。

基于 **Java 17** 的 AI 基础组件库，纯 ONNX Runtime 推理，零 Python / 零 PyTorch 依赖。专注 OCR、语音合成、语音识别、声纹识别等 AI 能力的集成与调用。

---

[✨✨✨推广：**BladeX 物联网平台**✨✨✨iot.bladex.cn](https://iot.bladex.cn?from=mica-mqtt)

## 模块导航

### 基础设施

| 模块 | 说明 |
|------|------|
| [mica-ai-common](mica-ai-common/README.md) | 公共模块：统一异常、ONNX Provider 工具、音频工具 |

### 核心能力（零 Spring 依赖）

| 模块 | 功能 | 说明 |
|------|------|------|
| [mica-ai-ppocr](mica-ai-core/mica-ai-ppocr/README.md) | PP-OCRv6 文字识别 | 检测 + 识别全链路，支持 tiny/small/medium 三种规格 |
| [mica-ai-tts](mica-ai-core/mica-ai-tts/README.md) | Kokoro TTS 语音合成 | 中英双语，103 个音色，可插拔 G2P 架构 |
| [mica-ai-voice](mica-ai-core/mica-ai-voice/README.md) | SenseVoice 语音识别 | 多语种识别，Trie 树热词雷达，长音频分段 |
| [mica-ai-speaker](mica-ai-core/mica-ai-speaker/README.md) | ERes2Net 声纹识别 | 声纹验证 / 说话人识别，256 维 Embedding |
| [mica-ai-intent](mica-ai-core/mica-ai-intent/README.md) | BERT 中文意图识别 | 按字分词 + Softmax 分类，兼容 HuggingFace 词表 |

### Spring Boot Starter（一行配置接入）

| Starter | 配置前缀 |
|---------|---------|
| [mica-ai-ppocr-spring-boot-starter](mica-ai-starters/mica-ai-ppocr-spring-boot-starter/README.md) | `mica.ai.ppocr` |
| [mica-ai-tts-spring-boot-starter](mica-ai-starters/mica-ai-tts-spring-boot-starter/README.md) | `mica.ai.tts` |
| [mica-ai-voice-spring-boot-starter](mica-ai-starters/mica-ai-voice-spring-boot-starter/README.md) | `mica.ai.voice` |
| [mica-ai-speaker-spring-boot-starter](mica-ai-starters/mica-ai-speaker-spring-boot-starter/README.md) | `mica.ai.speaker` |
| [mica-ai-intent-spring-boot-starter](mica-ai-starters/mica-ai-intent-spring-boot-starter/README.md) | `mica.ai.intent` |

---

## 项目结构

```
mica-ai/
├── mica-ai-common/           # 公共模块
├── mica-ai-core/             # 核心能力（聚合）
│   ├── mica-ai-ppocr/        #   OCR 文字识别
│   ├── mica-ai-tts/          #   TTS 语音合成
│   ├── mica-ai-voice/        #   语音识别
│   ├── mica-ai-speaker/      #   声纹识别
│   └── mica-ai-intent/       #   意图识别
└── mica-ai-starters/         # Spring Boot Starter（聚合）
    ├── mica-ai-ppocr-spring-boot-starter/
    ├── mica-ai-tts-spring-boot-starter/
    ├── mica-ai-voice-spring-boot-starter/
    ├── mica-ai-speaker-spring-boot-starter/
    └── mica-ai-intent-spring-boot-starter/
```

---

## 快速开始

### Maven 依赖

核心模块和 Starter 的 groupId 均为 `net.dreamlu`：

```xml
<!-- 以 TTS 为例：直接使用核心引擎 -->
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-tts</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<!-- 或使用 Spring Boot Starter（自动注入 Bean） -->
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-tts-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 最小示例

```java
// TTS 语音合成
KokoroTtsConfig config = KokoroTtsConfig.builder()
    .modelPath("model/model_dynamic.onnx")
    .voicesDir("model/voices")
    .configPath("model/config.json")
    .build();
try (KokoroTts tts = new KokoroTts(config)) {
    TtsResult result = tts.synthesize("你好世界！");
    tts.saveWav(result, "output.wav");
}
```

各模块详细用法请参阅对应 README。

---

## 设计原则

- **零 Spring 依赖** — 核心模块纯 Java，可在任何 Java 17+ 项目中独立使用
- **Builder 模式** — 所有引擎通过 Builder 风格 Config 类构造
- **AutoCloseable** — 引擎实现 `AutoCloseable`，支持 try-with-resources
- **可插拔** — 关键组件通过接口注入，支持自定义实现（如 TTS 的 G2P）
- **纯 ONNX Runtime** — 零 PyTorch / 零 Python 依赖，全 Java 推理

---

## 环境要求

| 组件 | 版本 |
|------|------|
| JDK | 17+ |
| Maven | 3.6+ |
| ONNX Runtime | 1.26.0 |

---

## 📄 License

本项目基于 Apache License 2.0 协议开源，详见 [LICENSE](LICENSE)。

### 🌟 致谢

感谢所有为 Mica 系列项目做出贡献的开发者。

### 💡 Mica AI 正在持续迭代中，欢迎 Star & Watch 关注最新动态！
