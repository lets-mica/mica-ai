<div align="center">

# 🚀 Mica AI

### 让 Java 工程师也能玩转主流 AI 模型 —— **零 Python · 零 PyTorch · 纯 ONNX Runtime**

[![Java](https://img.shields.io/badge/JDK-17%2B-orange?style=flat-square&logo=openjdk)](https://openjdk.org/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-4.1.0%2B-brightgreen?style=flat-square&logo=springboot)](https://spring.io/projects/spring-boot)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.26.0-blue?style=flat-square&logo=onnx)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Maven Central](https://img.shields.io/badge/Maven-2026.06.01-red?style=flat-square&logo=apache-maven)](https://mvnrepository.com/artifact/net.dreamlu/mica-ai)

> 一行依赖，五大 AI 能力开箱即用：OCR · TTS · ASR · 声纹 · 意图识别
>
> 给 Java 生态造的"AI 积木"，从此告别在 Java 里调 Python 微服务

[快速开始](#-快速开始) · [能力一览](#-六大能力一览) · [Spring Boot 一键接入](#-spring-boot-starter) · [应用场景](#-应用场景) · [WebSocket 实时识别](docs/websocket实时识别.md) · [更新日志](CHANGELOG.md)

</div>

---

## ✨ 为什么选 Mica AI？

还在为 Java 项目集成 AI 模型苦恼吗？

| 😩 痛点 | ✅ Mica AI 的解法 |
|--------|-----------------|
| Python 微服务部署运维成本高、跨语言调用调试难 | **纯 Java 推理**，JVM 里直接跑，无需任何 Python 进程 |
| PyTorch / PaddlePaddle 几百 MB 起步，包体爆炸 | **ONNX Runtime** 一个 runtime 全部搞定，CPU/GPU/CUDA/DML 自由切换 |
| 模型预处理、后处理各家一套，文档稀烂 | **端到端复刻 Python 实现**，预处理 / 后处理 / 解码全部内置，开箱即用 |
| 集成 Spring Boot 要写一堆 Bean 配置 | **官方 Starter**，一行 YAML 即可注入引擎 Bean |
| 国产模型下载速度感人 | **ModelScope 国内镜像优先**，配套 Python 工具链 `make download` 一键搞定 |

> 💡 **Mica AI 不是又一个 SDK，而是 Java 工程师的 AI 全家桶。**

---

## 🎯 六大能力一览

### 📦 一图看懂核心模块

```
                        ┌─────────────────────────────────────┐
                        │           Spring Boot App           │
                        └──────────────┬──────────────────────┘
                                       │ @Autowired
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
   ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
   │  mica-ai-tts     │   │   mica-ai-voice      │   │  mica-ai-ppocr   │
   │  语音合成 🎤     │   │   语音识别 🎧        │   │  文字识别 📷     │
   │  Kokoro-82M      │   │   SenseVoice         │   │  PP-OCRv6        │
   │  103 种音色       │   │   多语种 / 热词雷达   │   │  tiny/small/md   │
   └──────────────────┘   └──────────────────────┘   └──────────────────┘
              │                        │                        │
              ▼                        ▼                        ▼
   ┌──────────────────┐   ┌──────────────────────┐   ┌──────────────────┐
   │ mica-ai-speaker  │   │   mica-ai-intent     │   │   mica-ai-face   │
   │ 声纹识别 👤      │   │   中文意图识别 🧠    │   │  人脸识别 🎭     │
   │ ERes2Net 256 维  │   │   BERT 中文分类      │   │  InsightFace     │
   │                  │   │                      │   │  检测 + 512d 向量 │
   └──────────────────┘   └──────────────────────┘   └──────────────────┘
                                       │
                                       ▼
                          ┌──────────────────────┐
                          │     mica-ai-common   │
                          │  ONNX Provider       │
                          │  统一异常 / 音频      │
                          └──────────────────────┘
```

### 🔥 各能力速览

| 模块 | 能力 | 亮点 | 状态 |
|------|------|------|------|
| 🎤 [**mica-ai-tts**](mica-ai-core/mica-ai-tts/README.md) | Kokoro-82M 语音合成 | 中英双语 · 103 个音色 · 可插拔 G2P（默认 / houbb-pinyin / 自定义） | ✅ Stable |
| 🎧 [**mica-ai-voice**](mica-ai-core/mica-ai-voice/README.md) | SenseVoice 语音识别 | 多语种（中/英/日/韩/粤）· **Trie 树热词雷达** · 长音频自动分段 | ✅ Stable |
| 📷 [**mica-ai-ppocr**](mica-ai-core/mica-ai-ppocr/README.md) | PP-OCRv6 文字识别 | 检测+识别全链路 · tiny/small/medium 三档可选 · CPU bit-exact | ✅ Stable |
| 👤 [**mica-ai-speaker**](mica-ai-core/mica-ai-speaker/README.md) | ERes2Net 声纹识别 | 256 维 Embedding · 验证 / 识别双模式 · 80 维 FBank 特征 | ✅ Stable |
| 🧠 [**mica-ai-intent**](mica-ai-core/mica-ai-intent/README.md) | BERT 中文意图识别 | 按字分词 · Softmax 分类 · 兼容 HuggingFace 词表 | ✅ Stable |
| 🎭 [**mica-ai-face**](mica-ai-core/mica-ai-face/README.md) | InsightFace 人脸识别 | 检测 + 5 关键点对齐 + 512d 向量 · buffalo_l 精度优先 · **检索由向量库负责** | ✅ Stable |

---

## 🚀 快速开始

### 1️⃣ 添加 Maven 依赖

```xml
<!-- ① 直接用核心引擎（零 Spring） -->
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-tts</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<!-- ② 或使用 Spring Boot Starter（自动注入 Bean） -->
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-tts-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

> 💡 把 `mica-ai-tts` 换成 `mica-ai-voice` / `mica-ai-ppocr` / `mica-ai-speaker` / `mica-ai-intent` / `mica-ai-face` 即可。

### 2️⃣ 30 秒跑通一个 TTS

```java
KokoroTtsConfig config = KokoroTtsConfig.builder()
    .modelPath("model/model_dynamic.onnx")
    .voicesDir("model/voices")
    .configPath("model/config.json")
    .defaultVoice("zf_001")  // 中文女声
    .build();

try (KokoroTts tts = new KokoroTts(config)) {
    TtsResult result = tts.synthesize("你好，世界！欢迎使用 Mica AI 🎉");
    tts.saveWav(result, "hello.wav");  // 24kHz / 16-bit PCM
}
```

是的，就这么简单 —— 没有任何 Python 进程，没有任何中间服务。

### 3️⃣ 一行配置开启 Spring Boot Starter

```yaml
mica:
  ai:
    tts:
      model-path: model/model_dynamic.onnx
      voices-dir: model/voices
      config-path: model/config.json
      default-voice: zf_001
      onnx-provider: cpu   # 可选: cpu / cuda / dml
```

```java
@Service
@RequiredArgsConstructor
public class VoiceNotifyService {
    private final KokoroTts tts;   // ← 直接注入，无需任何 @Bean

    public void notify(String text) throws Exception {
        tts.saveWav(tts.synthesize(text), "notify.wav");
    }
}
```

---

## 🎬 各能力代码一瞥

### 📷 OCR 文字识别

```java
PPOcrV6Config config = PPOcrV6Config.builder()
    .detModelPath("models/PP-OCRv6_tiny_det_onnx/inference.onnx")
    .recModelPath("models/PP-OCRv6_tiny_rec_0515_onnx/inference.onnx")
    .recCharDictPath("models/rec_char_dict.txt")
    .build();

try (PPOcrV6Engine ocr = new PPOcrV6Engine(config)) {
    List<PPOcrV6Result> results = ocr.run(Imgcodecs.imread("card.png"));
    results.forEach(r -> System.out.printf("%s (%.3f)%n", r.text(), r.score()));
}
```

### 🎧 语音识别 + 热词雷达

```java
SenseVoiceConfig config = SenseVoiceConfig.builder()
    .encoderPath("model/SenseVoice-Encoder.fp32.onnx")
    .decoderPath("model/SenseVoice-CTC.fp32.onnx")
    .tokenizerPath("model/Tokenizer.bpe.model")
    .hotwords(List.of("Mica AI", "声纹识别", "Kokoro"))
    .build();

try (SenseVoice voice = new SenseVoice(config)) {
    TranscriptionResult result = voice.recognizeFile("meeting.wav");
    System.out.println("识别结果: " + result.text());
    System.out.println("命中热词: " + result.hotwords());
}
```

### 👤 声纹验证 / 说话人识别

```java
try (SpeakerVerifier verifier = new SpeakerVerifier("models/eres2net.onnx")) {
    float[] a = verifier.extractEmbedding("alice.wav");
    float[] b = verifier.extractEmbedding("alice2.wav");
    double sim = verifier.similarity(a, b);
    System.out.printf("声纹相似度: %.4f → 同一人: %s%n", sim, sim > 0.7);
}
```

### 🧠 中文意图识别

```java
BertIntentConfig config = BertIntentConfig.builder()
    .modelPath("bert_intent.onnx")
    .vocabPath("vocab.txt")
    .labels(List.of("weather", "music", "chat", "news"))
    .build();

try (BertIntent intent = new BertIntent(config)) {
    IntentResult r = intent.predict("今天天气怎么样");
    System.out.println("意图: " + r.intent() + ", 置信度: " + r.confidence());
}
```

> 完整 import 与 Starter 用法见 [mica-ai-intent/README.md](mica-ai-core/mica-ai-intent/README.md) 与 [mica-ai-intent-spring-boot-starter/README.md](mica-ai-starters/mica-ai-intent-spring-boot-starter/README.md)。

### 🎭 人脸识别（检测 + 512d 向量）

```java
try (FaceEngine face = FaceEngine.builder()
    .detModelPath(Path.of("models/det_10g.onnx"))
    .recModelPath(Path.of("models/w600k_r50.onnx"))
    .build()) {

    // 图片 → 所有人脸的 512d Embedding（已 L2 归一化）
    List<FaceEmbedding> faces = face.extract(Path.of("group.jpg"));

    // 入库 / 检索交给你自己的向量库（Milvus / pgvector）
    for (FaceEmbedding fe : faces) {
        milvusClient.insert("face_gallery", userId, fe.getVector());
    }
}
```

> mica-ai-face **只做检测 + 推理**，不做人脸库 / 1:N 检索。详见 [mica-ai-face/README.md](mica-ai-core/mica-ai-face/README.md)。

> 🌐 **WebSocket 实时识别**完整方案（含 VAD / 环形缓冲 / 流式推送）请看 [docs/websocket实时识别.md](docs/websocket实时识别.md)

---

## 🧰 Spring Boot Starter

每个能力都有对应的 Starter，遵循"**配置前缀** + **自动注入**"的 Spring Boot 约定：

| Starter | 配置前缀 | 一句话能力 |
|---------|---------|----------|
| [mica-ai-ppocr-spring-boot-starter](mica-ai-starters/mica-ai-ppocr-spring-boot-starter/README.md) | `mica.ai.ppocr` | OCR 文字识别 |
| [mica-ai-tts-spring-boot-starter](mica-ai-starters/mica-ai-tts-spring-boot-starter/README.md) | `mica.ai.tts` | TTS 语音合成 |
| [mica-ai-voice-spring-boot-starter](mica-ai-starters/mica-ai-voice-spring-boot-starter/README.md) | `mica.ai.voice` | ASR 语音识别 |
| [mica-ai-speaker-spring-boot-starter](mica-ai-starters/mica-ai-speaker-spring-boot-starter/README.md) | `mica.ai.speaker` | 声纹识别 |
| [mica-ai-intent-spring-boot-starter](mica-ai-starters/mica-ai-intent-spring-boot-starter/README.md) | `mica.ai.intent` | 中文意图识别 |
| [mica-ai-face-spring-boot-starter](mica-ai-starters/mica-ai-face-spring-boot-starter/README.md) | `mica.ai.face` | 人脸识别（512d 向量） |

只需在 `application.yml` 配好模型路径，对应 `Bean` 即可 `@Autowired` 直接用。

---

## 🏗️ 项目结构

```
mica-ai/
├── mica-ai-common/                       # 公共：ONNX Provider、统一异常、音频工具
├── mica-ai-core/                         # 核心引擎（零 Spring）
│   ├── mica-ai-ppocr/                    #   📷 PP-OCRv6
│   ├── mica-ai-tts/                      #   🎤 Kokoro TTS
│   ├── mica-ai-voice/                    #   🎧 SenseVoice
│   ├── mica-ai-speaker/                  #   👤 ERes2Net 声纹
│   ├── mica-ai-intent/                   #   🧠 BERT 意图
│   └── mica-ai-face/                     #   🎭 InsightFace 人脸
├── mica-ai-starters/                     # Spring Boot Starter
│   ├── mica-ai-ppocr-spring-boot-starter/
│   ├── mica-ai-tts-spring-boot-starter/
│   ├── mica-ai-voice-spring-boot-starter/
│   ├── mica-ai-speaker-spring-boot-starter/
│   ├── mica-ai-intent-spring-boot-starter/
│   └── mica-ai-face-spring-boot-starter/
├── model-tools/                          # Python 模型工具链（下载 / 转换 / 训练）
└── docs/                                 # 方案文档（WebSocket 实时识别、意图微调…）
```

---

## 🎨 设计原则

- 🎯 **零 Spring 依赖** — 核心模块纯 Java，可在任何 Java 17+ 项目中独立使用
- 🧱 **Builder 模式** — 所有引擎通过 `XxxConfig.builder()` 链式构造，类型安全、IDE 友好
- 🪶 **轻量 API** — 引擎实现 `AutoCloseable`，`try-with-resources` 一行管理资源
- 🔌 **可插拔架构** — 关键组件通过接口注入（如 TTS 的 `G2P`、语音的 `Tokenizer`）
- ⚡ **纯 ONNX Runtime** — 零 PyTorch / 零 PaddlePaddle / 零 Python 进程，JVM 内全栈推理
- 🌏 **国内友好** — 模型工具链默认走 ModelScope，国内下载速度拉满
- 🧪 **Bit-exact 优先** — 优先 CPU 一致性测试，需要 GPU 时换 `onnxruntime_gpu` 即可

---

## 🛠️ 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| ☕ JDK | **17+** | 推荐 Azul Zulu 17 / Temurin 17 |
| 📦 Maven | 3.6+ | 构建 / 打包 |
| 🧠 ONNX Runtime | 1.26.0 | Maven 自动拉取，CPU/GPU 可选 |
| 🌱 Spring Boot | 4.1.0+ | 仅 Starter 必需 |

> GPU 加速：把 `onnxruntime` 替换为 `onnxruntime_gpu`，并将 `onnxProvider` 设为 `cuda` / `dml` 即可。

---

## 🗺️ 应用场景

| 场景 | 推荐组合 |
|------|---------|
| 🏢 **智能客服 / 语音助手** | mica-ai-tts + mica-ai-voice + mica-ai-intent |
| 📝 **会议记录 / 字幕生成** | mica-ai-voice + WebSocket 实时方案 |
| 🔍 **票据 / 证件识别** | mica-ai-ppocr（tiny / small / medium） |
| 🔐 **声纹登录 / 反作弊** | mica-ai-speaker + 自建 Embedding 库 |
| 🎭 **人脸识别 / 门禁 / 考勤** | mica-ai-face + Milvus / pgvector（向量库做 1:N 检索） |
| 🤖 **IoT 语音交互** | mica-ai-voice（VAD）+ mica-ai-tts + mica-ai-intent |

> 📚 更多落地参考见 [docs/websocket实时识别.md](docs/websocket实时识别.md)（含完整 WebSocket + VAD 流式方案）

---

## 🤝 与 BladeX 物联网平台联动

[✨✨✨ **BladeX 物联网平台** ✨✨✨ iot.bladex.cn](https://iot.bladex.cn?from=mica-mqtt)

Mica AI 与 BladeX 物联网平台无缝集成，让 AI 能力直接落到边缘设备与物联网网关。

---

## 📄 License

本项目基于 [Apache License 2.0](LICENSE) 协议开源，可放心用于商业项目。

---

## 🌟 Star History

如果 Mica AI 对你有帮助，欢迎 ⭐ **Star** 支持一下，你的 star 是我们持续迭代的最大动力！

> 💡 **Mica AI 正在持续迭代中，欢迎 Star & Watch 关注最新动态！**

---

## 💖 致谢

感谢所有为 Mica 系列项目做出贡献的开发者，以及以下开源项目：

- [PP-OCRv6](https://github.com/PaddlePaddle/PaddleOCR) · [Kokoro TTS](https://github.com/hexgrad/kokoro) · [SenseVoice](https://github.com/FunASR/SenseVoice)
- [ERes2Net](https://github.com/speechbrain/ERes2Net) · [Chinese-BERT-WWM-Ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)
- [ONNX Runtime](https://onnxruntime.ai/) · [houbb/pinyin](https://github.com/houbb/pinyin) · [OpenCV](https://opencv.org/)

---

<div align="center">

**[⬆ 回到顶部](#-mica-ai)** · Made with ❤️ by [Mica Team](https://www.dreamlu.net)

</div>