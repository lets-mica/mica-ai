# mica-ai-tts-spring-boot-starter

> Spring Boot Starter for Kokoro TTS 语音合成，一行配置即可接入。

基于 [mica-ai-tts](../../mica-ai-core/mica-ai-tts/README.md) 核心引擎，提供 `@ConfigurationProperties` 自动配置与 `KokoroTts` Bean 自动注入。

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
    <artifactId>mica-ai-tts-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 2.2 配置文件

```yaml
mica:
  ai:
    tts:
      model-path: model/model_dynamic.onnx
      voices-dir: model/voices
      config-path: model/config.json
      default-voice: zf_001
      default-speed: 1.0
      onnx-provider: cpu
```

### 2.3 注入使用

```java
import net.dreamlu.mica.ai.tts.KokoroTts;
import net.dreamlu.mica.ai.tts.config.TtsResult;
import org.springframework.stereotype.Service;

@Service
public class TtsService {

    private final KokoroTts tts;

    public TtsService(KokoroTts tts) {
        this.tts = tts;
    }

    public void speak(String text) throws Exception {
        TtsResult result = tts.synthesize(text);
        tts.saveWav(result, "output.wav");
    }
}
```

---

## 3. 配置属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mica.ai.tts.model-path` | String | — | ONNX 模型路径（必填） |
| `mica.ai.tts.voices-dir` | String | — | 音色目录（必填） |
| `mica.ai.tts.config-path` | String | — | 词表配置 JSON 路径（必填） |
| `mica.ai.tts.default-voice` | String | `zf_001` | 默认音色 |
| `mica.ai.tts.default-speed` | float | `1.0` | 默认语速（0.5 ~ 2.0） |
| `mica.ai.tts.onnx-provider` | String | `cpu` | ONNX 执行提供者 |

### AutoConfiguration 条件

- `mica.ai.tts.enabled` 不为 `false`（默认启用）
- classpath 存在 `KokoroTts` 类
