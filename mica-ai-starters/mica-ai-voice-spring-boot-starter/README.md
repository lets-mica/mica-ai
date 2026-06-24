# mica-ai-voice-spring-boot-starter

> Spring Boot Starter for SenseVoice 语音识别，一行配置即可接入。

基于 [mica-ai-voice](../../mica-ai-core/mica-ai-voice/README.md) 核心引擎，提供 `@ConfigurationProperties` 自动配置与 `SenseVoice` Bean 自动注入。

---

## 1. 环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| JDK | 17+ | 推荐 Azul Zulu 17 / Temurin 17 / Oracle 17 |
| Spring Boot | 4.1.0+ | 自动配置依赖 |
| Maven | 3.6+ | 编译 / 打包 |

---

## 2. 快速使用

### 2.1 Maven 依赖

```xml
<dependency>
    <groupId>net.dreamlu</groupId>
    <artifactId>mica-ai-voice-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 2.2 配置文件

```yaml
mica:
  ai:
    voice:
      encoder-path: model/SenseVoice-Encoder.fp32.onnx
      decoder-path: model/SenseVoice-CTC.fp32.onnx
      tokenizer-path: model/Tokenizer.bpe.model
      onnx-provider: cpu
      top-k: 10
      itn: true
      hotwords:
        - 热词1
        - 热词2
```

### 2.3 注入使用

```java
import net.dreamlu.mica.ai.voice.config.TranscriptionResult;
import net.dreamlu.mica.ai.voice.engine.SenseVoice;
import org.springframework.stereotype.Service;

@Service
public class VoiceService {

    private final SenseVoice voice;

    public VoiceService(SenseVoice voice) {
        this.voice = voice;
    }

    public String transcribe(String wavPath) {
        TranscriptionResult result = voice.recognizeFile(wavPath);
        return result.text();
    }
}
```

---

## 3. 配置属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mica.ai.voice.encoder-path` | String | — | 编码器 ONNX 路径（必填） |
| `mica.ai.voice.decoder-path` | String | — | CTC 解码器 ONNX 路径（必填） |
| `mica.ai.voice.tokenizer-path` | String | — | SentencePiece 分词模型路径（必填） |
| `mica.ai.voice.onnx-provider` | String | `cpu` | ONNX 执行提供者 |
| `mica.ai.voice.top-k` | int | `10` | 热词搜索深度 |
| `mica.ai.voice.itn` | boolean | `true` | 中文数字逆文本规范化 |
| `mica.ai.voice.hotwords` | List\<String\> | — | 热词列表 |

### AutoConfiguration 条件

- `mica.ai.voice.enabled` 不为 `false`（默认启用）
- classpath 存在 `SenseVoice` 类
