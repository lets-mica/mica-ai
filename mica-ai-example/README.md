# mica-ai-example

> mica-ai Spring Boot 集成测试 / Demo：聚合 6 个 Starter 的可运行示例与 `@SpringBootTest`。

本模块面向集成测试场景：

1. **可运行的 Spring Boot Demo**：`mvn spring-boot:run` 启动后，可通过 REST 端点手工触发 TTS / ASR / OCR / 声纹 / 意图 / 人脸能力。
2. **`@SpringBootTest` 集成测试**：覆盖 6 个 Starter 的自动装配行为（`enabled` 开关、fail-fast、Bean 注入）。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Azul Zulu 17 / Temurin 17 |
| Maven | 3.6+ | 多模块构建 |
| Spring Boot | 4.1.0+ | 由根 BOM 引入 |
| ONNX 模型 | 视能力 | 见各子模块 README |

---

## 2. 快速开始

### 2.1 默认启动（所有能力 disabled）

```bash
# 编译
mvn -pl mica-ai-example -am -DskipTests clean install

# 直接启动（不会注入任何 Engine Bean）
mvn -pl mica-ai-example spring-boot:run
```

启动后访问 `http://localhost:8181`，由于没有任何能力启用，所有 Controller 都已被 `@ConditionalOnProperty(mica.ai.<cap>.enabled)` 跳过注册。

### 2.2 启用指定能力

编辑 [src/main/resources/application.yml](src/main/resources/application.yml)，取消对应能力的注释、设置 `enabled: true`、并把模型路径改成你的实际路径：

```yaml
mica:
  ai:
    tts:
      enabled: true
      model-path: E:/codes/ai/kokoro-onnx/model/model_dynamic.onnx
      voices-dir: E:/codes/ai/kokoro-onnx/model/voices
      config-path: E:/codes/ai/kokoro-onnx/model/config.json
```

> ⚠️ 当 `enabled=true` 但必填项（如 `model-path`）缺失时，应用启动会 **fail-fast** 抛出 `MicaAiException`。
> 不希望加载该能力时，直接设 `enabled: false` 即可。

### 2.3 跑集成测试

```bash
mvn -pl mica-ai-example -am test
```

测试覆盖：

- `DisabledAllTest`：默认配置下 ApplicationContext 能正常启动，6 个 Engine Bean 都不存在。
- `ttsEnabledButMissingRequiredShouldFailFast`：TTS `enabled=true` 但缺 `model-path` → 启动抛 `MicaAiException`。
- `intentEnabledButMissingLabelsShouldFailFast`：Intent `enabled=true` 但缺 `labels` → 启动抛 `MicaAiException`。
- `EnabledWithPlaceholderTest`：TTS `enabled=true` 且路径齐全 → `KokoroTts` Bean 注入成功。

> 集成测试不依赖任何真实 ONNX 模型文件，可放心在 CI 中执行。

---

## 3. REST 端点一览

| 路径 | 方法 | 说明 | 依赖 Starter |
|------|------|------|--------------|
| `/tts/voices` | GET | 列出可用音色 | mica-ai-tts |
| `/tts/synthesize?text=...` | POST | 文本合成语音（返回 WAV 字节流） | mica-ai-tts |
| `/tts/synthesize-from-phonemes` | POST | 用预生成音素合成（绕过 G2P） | mica-ai-tts |
| `/voice/recognize` | POST | 上传 WAV，返回识别文本 + 时间戳 + 热词 | mica-ai-voice |
| `/voice/hotwords` | PUT | 动态更新热词列表 | mica-ai-voice |
| `/ppocr/recognize` | POST | 上传图片，返回识别到的文本行 | mica-ai-ppocr |
| `/speaker/enroll` | POST | 上传多段 WAV，返回 192 维 embedding | mica-ai-speaker |
| `/speaker/verify` | POST | 上传 enroll + test，返回 cosine 相似度 | mica-ai-speaker |
| `/intent/predict` | POST | 单条中文文本意图分类 | mica-ai-intent |
| `/intent/predict-batch` | POST | 批量意图分类 | mica-ai-intent |
| `/face/detect` | POST | 上传图片，返回人脸框 + 关键点 | mica-ai-face |
| `/face/extract` | POST | 上传图片，返回 512 维 embedding | mica-ai-face |

所有 Controller 都用 `@ConditionalOnProperty(prefix = "mica.ai.<cap>", name = "enabled")` 装饰，未启用对应能力时不会注册，访问会返回 404。

---

## 4. 项目结构

```
mica-ai-example/
├── pom.xml                                          # 依赖 6 个 starter + spring-boot-starter-web
├── src/main/java/net/dreamlu/mica/ai/example/
│   ├── ExampleApplication.java                      # @SpringBootApplication 入口
│   └── controller/
│       ├── TtsController.java
│       ├── VoiceController.java
│       ├── PpocrController.java
│       ├── SpeakerController.java
│       ├── IntentController.java
│       └── FaceController.java
├── src/main/resources/
│   ├── application.yml                              # 全部 disabled + 启用示例
│   └── logback-spring.xml
└── src/test/java/net/dreamlu/mica/ai/example/
    └── ExampleApplicationContextTest.java           # @SpringBootTest 集成测试
```

---

## 5. 已知约束

- **OpenCV 原生库**：仅 `mica-ai-ppocr` 内部重度依赖 `org.opencv.*`（PP-OCRv6 文本框后处理），
  `OpenCVNativeLoader` 已由 `mica-ai-ppocr-spring-boot-starter` 以 `@AutoConfiguration` 形式自动注入，
  并通过 `@AutoConfigureBefore(PPOCRAutoConfiguration.class)` 保证在 PP-OCR Engine 初始化之前完成 native 加载。
  `mica-ai-face` 完全不依赖 openpnp/opencv（仅用 ONNX Runtime 跑 YuNet+SFace）。
- **TTS 缺 G2P 依赖**：默认 `ChineseG2P` 简化实现。要用 houbb/pinyin 多音字 G2P，
  请在 `mica-ai-example/pom.xml` 显式追加 `com.github.houbb:pinyin` 依赖。
- **multipart 上传**：默认最大 50MB / 请求 100MB，可在 `application.yml` 中调整 `spring.servlet.multipart`。