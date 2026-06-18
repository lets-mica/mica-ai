# mica-ai-speaker-spring-boot-starter

> Spring Boot Starter for ERes2Net 声纹识别，一行配置即可接入。

基于 [mica-ai-speaker](../../mica-ai-core/mica-ai-speaker/README.md) 核心引擎，提供 `@ConfigurationProperties` 自动配置与 `SpeakerVerifier` Bean 自动注入。

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
    <artifactId>mica-ai-speaker-spring-boot-starter</artifactId>
    <version>${mica-ai.version}</version>
</dependency>
```

### 2.2 配置文件

```yaml
mica:
  ai:
    speaker:
      model-path: models/eres2net.onnx
      onnx-provider: cpu
```

### 2.3 注入使用

```java
import net.dreamlu.mica.ai.speaker.SpeakerVerifier;
import org.springframework.stereotype.Service;

@Service
public class SpeakerService {

    private final SpeakerVerifier verifier;

    public SpeakerService(SpeakerVerifier verifier) {
        this.verifier = verifier;
    }

    public double verify(String enrollPath, String testPath) {
        return verifier.verify(enrollPath, testPath);
    }

    public float[] extractEmbedding(String audioPath) {
        return verifier.extractEmbedding(audioPath);
    }
}
```

---

## 3. 配置属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mica.ai.speaker.model-path` | String | — | ONNX 模型路径（必填） |
| `mica.ai.speaker.onnx-provider` | String | `cpu` | ONNX 执行提供者 |

### AutoConfiguration 条件

- `mica.ai.speaker.enabled` 不为 `false`（默认启用）
- classpath 存在 `SpeakerVerifier` 类
