<div align="center">

# 🚀 Mica AI

### 让 Java 工程师也能玩转主流 AI 模型 —— **零 Python · 零 PyTorch · 纯 ONNX Runtime**

[![Java](https://img.shields.io/badge/JDK-8%2B-orange?style=flat-square&logo=openjdk)](https://openjdk.org/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-2.7.x-brightgreen?style=flat-square&logo=springboot)](https://spring.io/projects/spring-boot)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.18.0-blue?style=flat-square&logo=onnx)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-red?style=flat-square&logo=opencv)](https://github.com/openpnp/openpnp-vision)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Maven Central](https://img.shields.io/badge/Maven-1.0.0-red?style=flat-square&logo=apache-maven)](https://mvnrepository.com/artifact/net.dreamlu/mica-ai)

> 一行依赖，人脸检测 + 128d Embedding + 活体 + 头像 / 证件卡片提取 开箱即用
>
> 给 Java 生态造的"AI 积木"，从此告别在 Java 里调 Python 微服务

[快速开始](#-快速开始) · [Spring Boot 一键接入](#-spring-boot-starter) · [应用场景](#-应用场景) · [更新日志](CHANGELOG.md)

</div>

---

## ✨ 为什么选 Mica AI？

还在为 Java 项目集成 AI 模型苦恼吗？

| 😩 痛点 | ✅ Mica AI 的解法 |
|--------|-----------------|
| Python 微服务部署运维成本高、跨语言调用调试难 | **纯 Java 推理**，JVM 里直接跑，无需任何 Python 进程 |
| PyTorch / PaddlePaddle 几百 MB 起步，包体爆炸 | **ONNX Runtime** 一个 runtime 全部搞定，CPU/GPU/CUDA 自由切换 |
| 模型预处理、后处理各家一套，文档稀烂 | **端到端复刻 OpenCV C++ 实现**，预处理 / 后处理 / 解码全部内置，开箱即用 |
| 集成 Spring Boot 要写一堆 Bean 配置 | **官方 Starter**，一行 YAML 即可注入引擎 Bean |
| 国产模型下载速度感人 | **ModelScope 国内镜像优先**，配套 Python 工具链 `make download` 一键搞定 |

> 💡 **Mica AI 不是又一个 SDK，而是 Java 工程师的 AI 全家桶。**

---

## 🎯 当前能力一览

```
                        ┌─────────────────────────────────────┐
                        │           Spring Boot App           │
                        └──────────────┬──────────────────────┘
                                       │ @Autowired
                                       ▼
                          ┌──────────────────────┐
                          │     mica-ai-face     │
                          │  人脸识别 🎭         │
                          │  YuNet + SFace       │
                          │  + 活体 MiniFASNet   │
                          │  + 头像 / 证件卡片   │
                          └──────────────────────┘
                                       │
                                       ▼
                          ┌──────────────────────┐
                          │    mica-ai-common    │
                          │  ONNX 通用基础设施    │
                          │  + 统一异常          │
                          └──────────────────────┘
```

能力清单：

| 模块 | 模型 | 维度 | License |
|------|------|------|---------|
| 检测 | YuNet (`face_detection_yunet_2023mar.onnx`) | — | Apache 2.0 |
| 特征 | SFace (`face_recognition_sface_2021dec.onnx`) | 128d | Apache 2.0 |
| 活体 | MiniFASNetV2 (`2.7_80x80_MiniFASNetV2.onnx`) | 3 类 | MIT（minivision Silent-Face-Anti-Spoofing，可商用） |

> 📦 音频（TTS / ASR / 声纹）和 OCR / 意图识别能力已抽离到独立的 mica-* 项目，本仓库只保留人脸相关模块。
>
> - OCR：[**mica-ppocr**](https://gitee.com/dreamlu/mica-ppocr) — PaddleOCR / PP-OCRv4 的 Java 推理
> - 语音（ASR / 热词雷达 / 中文 ITN）：[**mica-voice**](https://gitee.com/dreamlu/mica-voice) — SenseVoice 等语音模型的 Java 推理
>
> 📌 活体模型默认**关闭**（`mica.ai.face.liveness.enabled=true` 显式启用），无 liveness 模型路径时跳过加载；商业落地前请按 §6.1 自查模型许可（当前 MiniFASNetV2 已确认 MIT，可商用）。

---

## 🚀 快速开始

### 1️⃣ 添加 Maven 依赖

```xml
<!-- ① 直接用核心引擎（零 Spring） -->
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-face</artifactId>
    <version>${mica-ai.version}</version>
</dependency>

<!-- ② 或使用 Spring Boot Starter（自动注入 Bean） -->
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-face-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 2️⃣ 30 秒跑通一个人脸识别

```java
ModelConfig config = ModelConfig.builder()
    .detectionModelPath("models/face_detection_yunet_2023mar.onnx")
    .recognitionModelPath("models/face_recognition_sface_2021dec.onnx")
    .build();

try (ModelManager manager = ModelManager.create(config)) {
    FaceDetector detector = new FaceDetector(manager);
    FaceAligner aligner = new FaceAligner();
    FeatureExtractor extractor = new FeatureExtractor(manager);

    Mat img = ImageUtils.byteArrayToMat(Files.readAllBytes(Path.of("group.jpg")));
    List<FaceBox> boxes = detector.detect(img);
    for (FaceBox box : boxes) {
        try (Mat aligned = aligner.align(img, box)) {
            float[] feature = extractor.extract(aligned);   // 128d L2 归一化
            // 入库 / 检索交给你自己的向量库（Milvus / pgvector）
        }
    }
}
```

### 3️⃣ 一行配置开启 Spring Boot Starter

```yaml
mica:
  ai:
    face:
      model:
        detection:
          path: classpath:models/face_detection_yunet_2023mar.onnx
        recognition:
          path: classpath:models/face_recognition_sface_2021dec.onnx
        liveness:
          path: classpath:models/2.7_80x80_MiniFASNetV2.onnx
      detection:
        threshold: 0.9
        nms-threshold: 0.3
      verify:
        threshold: 0.35
      avatar:
        size: 256
      card:
        output-width: 1011
        output-height: 638
      device: cpu
      onnx:
        intra-op-num-threads: 0
        graph-optimization-level: ORT_ENABLE_ALL
```

```java
@Service
@RequiredArgsConstructor
public class FaceEnrollService {
    private final FaceDetector detector;       // ← 直接注入
    private final FaceAligner aligner;
    private final FeatureExtractor extractor;
    private final AvatarExtractor avatarExtractor;  // 头像提取
    private final CardExtractor cardExtractor;      // 证件卡片提取
    private final FaceVerifier verifier;            // 1:1 比对
}
```

---

## 🧰 Spring Boot Starter

| Starter | 配置前缀 | 一句话能力 |
|---------|---------|----------|
| [mica-ai-face-spring-boot-starter](mica-ai-starters/mica-ai-face-spring-boot-starter/README.md) | `mica.ai.face` | 人脸检测 + 128d 特征 + 活体 + 头像 / 证件卡片提取 |

只需在 `application.yml` 配好模型路径，对应 `Bean` 即可 `@Autowired` 直接用。

---

## 🏗️ 项目结构

```
mica-ai/
├── mica-ai-common/                       # 公共：ONNX 通用基础设施、统一异常
├── mica-ai-core/                         # 核心引擎（零 Spring，纯 Java 8）
│   └── mica-ai-face/                     #   🎭 OpenCV Zoo 人脸识别
├── mica-ai-starters/                     # Spring Boot 2 Starter
│   └── mica-ai-face-spring-boot-starter/
├── mica-ai-example/                      # Spring Boot 集成示例
└── model-tools/                          # Python 模型工具链（下载 / 转换）
    ├── common/                           #   downloader、onnx_utils、progress
    ├── face/                             #   face 能力脚本
    └── scripts/                          #   smoke_test / publish / package
```

---

## 🎨 设计原则

- 🎯 **零 Spring 依赖** — 核心模块纯 Java，可在任何 Java 8+ 项目中独立使用
- 🧱 **Builder 模式** — 所有引擎通过 `XxxConfig.builder()` 链式构造，类型安全、IDE 友好
- 🪶 **轻量 API** — 引擎实现 `AutoCloseable` / `@PreDestroy`，`try-with-resources` 一行管理资源
- ⚡ **纯 ONNX Runtime** — 零 PyTorch / 零 PaddlePaddle / 零 Python 进程，JVM 内全栈推理
- 🌏 **国内友好** — 模型工具链默认走 ModelScope，国内下载速度拉满
- 🧪 **Bit-exact 优先** — 优先 CPU 一致性测试，需要 GPU 时换 `onnxruntime_gpu` 即可

---

## 🛠️ 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| ☕ JDK | **8+** | 推荐 Temurin / Azul Zulu 8、11、17 |
| 📦 Maven | 3.6+ | 构建 / 打包 |
| 🧠 ONNX Runtime | 1.18.0 | Maven 自动拉取，CPU/GPU 可选 |
| 🖼️ OpenCV | 4.9.0（openpnp） | Maven 自动拉取对应系统 / 架构的原生库 |

> GPU 加速：把 `onnxruntime` 替换为 `onnxruntime_gpu`，并将 `mica.ai.face.device=gpu`（需 CUDA Toolkit + 驱动）。

---

## 🗺️ 应用场景

| 场景 | 推荐组合 |
|------|---------|
| 🎭 **人脸识别 / 门禁 / 考勤** | mica-ai-face + Milvus / pgvector（向量库做 1:N 检索） |
| 🪪 **证件核验 / 人证合一** | mica-ai-face.FaceVerifier + CardExtractor |
| 🖼️ **头像 / 证件卡片标准化** | mica-ai-face.AvatarExtractor / CardExtractor |

---

## 📄 License

本项目基于 [Apache License 2.0](LICENSE) 协议开源，可放心用于商业项目。

---

## 💖 致谢

感谢所有为 Mica 系列项目做出贡献的开发者，以及以下开源项目：

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) · [ONNX Runtime](https://onnxruntime.ai/) · [openpnp/openpnp-vision](https://github.com/openpnp/openpnp-vision)
- 已抽离的 Mica 系列仓库：[mica-ppocr](https://gitee.com/dreamlu/mica-ppocr) · [mica-voice](https://gitee.com/dreamlu/mica-voice)

<div align="center">

**[⬆ 回到顶部](#-mica-ai)** · Made with ❤️ by [Mica Team](https://www.dreamlu.net)

</div>