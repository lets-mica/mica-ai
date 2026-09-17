<div align="center">

# 🚀 Mica AI

### 让 Java 工程师也能玩转主流 AI 模型 —— **零 Python · 零 PyTorch · 纯 ONNX Runtime**

[![Java](https://img.shields.io/badge/JDK-8%2B-orange?style=flat-square&logo=openjdk)](https://openjdk.org/)
[![Spring Boot](https://img.shields.io/badge/Spring%20Boot-2.7.x~4.x-brightgreen?style=flat-square&logo=springboot)](https://spring.io/projects/spring-boot)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.18.0-blue?style=flat-square&logo=onnx)](https://onnxruntime.ai/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-red?style=flat-square&logo=opencv)](https://github.com/openpnp/openpnp-vision)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Maven Central](https://img.shields.io/badge/Maven-1.0.0-red?style=flat-square&logo=apache-maven)](https://mvnrepository.com/artifact/net.dreamlu/mica-ai)

> 一行依赖，人脸检测 + 128d Embedding + 活体 + 头像 / 证件卡片提取 + 文件类型识别 + 中国车牌识别 开箱即用
>
> 给 Java 生态造的"AI 积木"，从此告别在 Java 里调 Python 微服务

[快速开始](#-快速开始) · [Spring Boot Starter](#-spring-boot-starter) · [应用场景](#-应用场景) · [更新日志](CHANGELOG.md)

</div>

---

## ✨ 为什么选 Mica AI？

还在为 Java 项目集成 AI 模型苦恼吗？

| 😩 痛点 | ✅ Mica AI 的解法 |
|--------|-----------------|
| Python 微服务部署运维成本高、跨语言调用调试难 | **纯 Java 推理**，JVM 里直接跑，无需任何 Python 进程 |
| PyTorch / PaddlePaddle 几百 MB 起步，包体爆炸 | **ONNX Runtime** 一个 runtime 全部搞定，CPU/GPU/CUDA 自由切换 |
| 模型预处理、后处理各家一套，文档稀烂 | **端到端复刻 OpenCV / Magika / HyperLPR3 实现**，预处理 / 后处理 / 解码全部内置，开箱即用 |
| 集成 Spring Boot 要写一堆 Bean 配置 | **官方 Starter**，一行 YAML 注入引擎 Bean |
| 模型下载慢、上手要跑 Python 脚本 | **模型直接入库**（均 <50MB），克隆即用，`make -C model-tools smoke` 一键自检 |

> 💡 **Mica AI 不是又一个 SDK，而是 Java 工程师的 AI 全家桶。**

---

## 🎯 当前能力一览

| 能力 | 模块 | 核心模型 | 输出维度 / 类别 | License | 商用 |
|------|------|---------|----------------|---------|------|
| 🎭 人脸检测 | `mica-ai-face` | YuNet | 人脸框 + 5 关键点 | Apache 2.0 | ✅ |
| 🧬 人脸特征 | `mica-ai-face` | SFace | **128d** L2 归一化向量 | Apache 2.0 | ✅ |
| 🛡️ 活体检测 | `mica-ai-face` | MiniFASNetV2 | real / replay / print | MIT | ✅ |
| 🖼️ 头像 / 证件卡片提取 | `mica-ai-face` | 几何变换 + USM/CLAHE | 正方形头像 / 矫正卡面 | Apache 2.0 | ✅ |
| 📄 文件类型识别 | `mica-ai-filetype` | Google Magika `standard_v3_3` | **214 类** + mime / group / description | Apache 2.0 | ✅ |
| 🚗 中国车牌识别 | `mica-ai-plate` | HyperLPR3 v20230229 | 车牌号 + 10 类判型 + 颜色 | Apache 2.0 | ✅ |

### 整体架构

```
                    ┌─────────────────────────────────────┐
                    │           Spring Boot App           │
                    └──────────────┬──────────────────────┘
                                   │ @Autowired / 一行 YAML
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │                    mica-ai-starters                  │
        │  face-starter 🎭   filetype-starter 📄  plate-starter 🚗│
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │                     mica-ai-core                     │
        │   mica-ai-face        mica-ai-filetype  mica-ai-plate│
        │   YuNet + SFace 🎭    Magika 214 类 📄  HyperLPR3 🚗  │
        │   + 活体 MiniFASNet   + mime / group   + 车牌号 / 颜色│
        │   + 头像 / 证件卡片                                  │
        └──────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                          ┌──────────────────────┐
                          │    mica-ai-common    │
                          │  ONNX 通用基础设施    │
                          │  + 统一异常          │
                          └──────────────────────┘
```

> 🔌 零 Spring 场景可直接依赖 `mica-ai-core` 各模块，Starter 仅是 Bean 注入的便捷封装；`mica-ai-example` 提供完整集成示例。

> - OCR：[**mica-ppocr**](https://gitee.com/dreamlu/mica-ppocr) — PaddleOCR / PP-OCRv4 的 Java 推理
> - 语音（ASR / TTS / 热词雷达 / 中文 ITN）：[**mica-voice**](https://gitee.com/dreamlu/mica-voice) — SenseVoice 等语音模型的 Java 推理

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

其他能力依赖：

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-filetype</artifactId>          <!-- 文件类型识别 -->
    <version>${mica-ai.version}</version>
</dependency>
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-plate</artifactId>             <!-- 中国车牌识别 -->
    <version>${mica-ai.version}</version>
</dependency>
```

### 2️⃣ 30 秒跑通一个人脸识别（纯 Java）

```java
ModelConfig config = ModelConfig.builder()
    .detectionModelPath(Path.of("models/face_detection_yunet_2023mar.onnx"))
    .recognitionModelPath(Path.of("models/face_recognition_sface_2021dec.onnx"))
    .build();

try (ModelManager manager = ModelManager.create(config)) {
    FaceDetector detector = new FaceDetector(manager);
    FaceAligner aligner   = new FaceAligner();
    FeatureExtractor extractor = new FeatureExtractor(manager);

    BufferedImage img = ImageIO.read(new File("group.jpg"));
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
      enabled: true
      device: cpu
      detection:
        model-path: classpath:models/face_detection_yunet_2023mar.onnx
        threshold: 0.9
        nms-threshold: 0.3
      recognition:
        model-path: classpath:models/face_recognition_sface_2021dec.onnx
      liveness:
        enabled: false
        model-path: classpath:models/2.7_80x80_MiniFASNetV2.onnx
      verify:
        threshold: 0.35
      avatar:
        size: 256
      card:
        output-width: 1011
        output-height: 638
      onnx:
        intra-op-num-threads: 0
        graph-optimization-level: ORT_ENABLE_ALL
```

```java
@Service
@RequiredArgsConstructor
public class FaceEnrollService {
    private final FaceDetector detector;            // ← 直接注入
    private final FaceAligner aligner;
    private final FeatureExtractor extractor;       // 128d
    private final LivenessDetector liveness;        // 活体（启用后可用）
    private final FaceVerifier verifier;            // 1:1 比对
    private final AvatarExtractor avatarExtractor;  // 头像提取
    private final CardExtractor cardExtractor;      // 证件卡片提取
}
```

---

## 🧰 Spring Boot Starter

| Starter | 配置前缀 | 一句话能力 |
|---------|---------|----------|
| [mica-ai-face-spring-boot-starter](mica-ai-starters/mica-ai-face-spring-boot-starter/README.md) | `mica.ai.face` | 人脸检测 + 128d 特征 + 活体 + 头像 / 证件卡片提取 |
| [mica-ai-filetype-spring-boot-starter](mica-ai-starters/mica-ai-filetype-spring-boot-starter/README.md) | `mica.ai.filetype` | Google Magika 复刻，214 类文件类型识别（含 / 排除置信度三模式） |
| [mica-ai-plate-spring-boot-starter](mica-ai-starters/mica-ai-plate-spring-boot-starter/README.md) | `mica.ai.plate` | HyperLPR3 中国车牌识别（检测 + CRNN 识别 + 颜色分类 + 10 类判型） |

只需在 `application.yml` 配好模型路径，对应 Bean 即可 `@Autowired` 直接用。

---

## 🏗️ 项目结构

```
mica-ai/
├── pom.xml                         # 顶层 BOM（revision / spring-boot / onnxruntime）
├── mica-ai-common/                 # 公共：ONNX 通用基础设施、统一异常
├── mica-ai-core/                   # 核心引擎（零 Spring，纯 Java 8+）
│   ├── mica-ai-face/               #   🎭 OpenCV Zoo 人脸识别
│   ├── mica-ai-filetype/           #   📄 Google Magika 文件类型识别
│   └── mica-ai-plate/              #   🚗 HyperLPR3 中国车牌识别
├── mica-ai-starters/               # Spring Boot 2 Starter
│   ├── mica-ai-face-spring-boot-starter/
│   ├── mica-ai-filetype-spring-boot-starter/
│   └── mica-ai-plate-spring-boot-starter/
├── mica-ai-example/                # Spring Boot 集成示例
└── model-tools/                    # 模型资产（直接入库，均 <50MB）
    ├── face/models/                #   YuNet + SFace + MiniFASNetV2
    ├── filetype/models/            #   Magika standard_v3_3
    └── plate/models/               #   HyperLPR3 v20230229
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

> GPU 加速：把 `onnxruntime` 替换为 `onnxruntime_gpu`，并将 `device=gpu`（需 CUDA Toolkit + 驱动）。

---

## 🗺️ 应用场景

| 场景 | 推荐组合 |
|------|---------|
| 🎭 **人脸识别 / 门禁 / 考勤** | `mica-ai-face` + Milvus / pgvector（向量库做 1:N 检索） |
| 🪪 **证件核验 / 人证合一** | `mica-ai-face` 的 `FaceVerifier` + `CardExtractor` |
| 🖼️ **头像 / 证件卡片标准化** | `mica-ai-face` 的 `AvatarExtractor` / `CardExtractor` |
| 📄 **任意文件 MIME 推断 / 内容审计** | `mica-ai-filetype` — 214 类 + 三种置信度模式 |
| 🚗 **停车场 / 道闸 / 智慧出行** | `mica-ai-plate` 的 `PlatePipeline` + 颜色分类兜底 |

---

## 📄 License

本项目基于 [Apache License 2.0](LICENSE) 协议开源，可放心用于商业项目；当前所有依赖模型均确认可商用（详见 `AGENTS.md` §6.1）。

---

## 💖 致谢

感谢所有为 Mica 系列项目做出贡献的开发者，以及以下开源项目：

- [OpenCV Zoo](https://github.com/opencv/opencv_zoo) · [minivision Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) · [ONNX Runtime](https://onnxruntime.ai/) · [openpnp/openpnp-vision](https://github.com/openpnp/openpnp-vision) · [Google Magika](https://github.com/google/magika) · [HyperLPR3](https://github.com/szad670401/HyperLPR)
- 已抽离的 Mica 系列仓库：[mica-ppocr](https://gitee.com/dreamlu/mica-ppocr) · [mica-voice](https://gitee.com/dreamlu/mica-voice)

<div align="center">

**[⬆ 回到顶部](#-mica-ai)** · Made with ❤️ by [Mica Team](https://www.dreamlu.net)

</div>